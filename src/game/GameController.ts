import { Chess } from 'chess.js';
import type { Square } from 'chess.js';
import { ChessBoard } from '../board/ChessBoard';
import type { EngineResult } from '../engine/search';

export type PlayerType = 'human' | 'engine';
export type GameMode = 'human-human' | 'human-engine' | 'engine-engine';

interface GameConfig {
  mode: GameMode;
  engineSide: 'w' | 'b';
  timeControl: number;    // seconds per side
  increment: number;      // seconds increment
  engineDepth: number;    // max search depth
  engineTimeMs: number;   // ms per move for engine
}

interface PlayerState {
  name: string;
  timeRemainingMs: number;
}

export class GameController {
  private chess: Chess;
  private board: ChessBoard;
  private config: GameConfig;
  private players: { w: PlayerState; b: PlayerState };
  private clockInterval: ReturnType<typeof setInterval> | null = null;
  private engineRunning = false;
  private gameOver = false;
  private moveHistory: { san: string; lan: string; color: 'w' | 'b' }[] = [];
  private worker: Worker | null = null;
  private engineMode: 'ab' | 'mcts' | 'nn' = 'ab';

  // Pending promotion
  private pendingPromoFrom: Square | null = null;
  private pendingPromoTo: Square | null = null;

  // Callbacks
  onClockUpdate?: (w: number, b: number) => void;
  onMoveHistoryUpdate?: (moves: typeof this.moveHistory) => void;
  onEngineInfo?: (info: EngineResult) => void;
  onGameOver?: (result: string, reason: string) => void;
  onPromotionNeeded?: (from: Square, to: Square) => void;
  onScoreUpdate?: (score: number) => void;
  onCapturesUpdate?: (wCaptures: string[], bCaptures: string[]) => void;
  onEngineThinking?: (thinking: boolean) => void;

  constructor(boardEl: HTMLElement, config: Partial<GameConfig> = {}) {
    this.chess = new Chess();
    this.config = {
      mode: 'human-engine',
      engineSide: 'b',
      timeControl: 10 * 60,
      increment: 0,
      engineDepth: 20,
      engineTimeMs: 2000,
      ...config,
    };

    this.players = {
      w: { name: this.config.engineSide === 'w' ? 'Engine' : 'You', timeRemainingMs: this.config.timeControl * 1000 },
      b: { name: this.config.engineSide === 'b' ? 'Engine' : 'You', timeRemainingMs: this.config.timeControl * 1000 },
    };

    this.board = new ChessBoard(boardEl, this.chess);
    this.board.onMove((from, to, promo) => this.handleHumanMove(from, to, promo));
    this.board.onPromotion((from, to) => this.handlePromotionNeeded(from, to));

    this.initWorker();
    this.board.setEnabled(this.isHumanTurn());
    this.startClock();

    if (!this.isHumanTurn()) {
      setTimeout(() => this.runEngine(), 400);
    }
  }

  // ── Web Worker ────────────────────────────────────────────────────────────

  private initWorker() {
    // Terminate previous worker if any
    this.worker?.terminate();
    // Vite handles bundling this as a separate worker chunk
    this.worker = new Worker(
      new URL('../engine/worker.ts', import.meta.url),
      { type: 'module' }
    );
  }

  private runEngineInWorker(fen: string, timeLimitMs: number, maxDepth: number): Promise<EngineResult> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error('No worker'));
        return;
      }

      const onMessage = (e: MessageEvent<EngineResult>) => {
        this.worker!.removeEventListener('message', onMessage);
        this.worker!.removeEventListener('error', onError);
        resolve(e.data);
      };

      const onError = (e: ErrorEvent) => {
        this.worker!.removeEventListener('message', onMessage);
        this.worker!.removeEventListener('error', onError);
        reject(e);
      };

      this.worker.addEventListener('message', onMessage);
      this.worker.addEventListener('error', onError);
      this.worker.postMessage({ fen, timeLimitMs, maxDepth, mode: this.engineMode });

    });
  }

  // ── New Game ──────────────────────────────────────────────────────────────

  newGame(config?: Partial<GameConfig>) {
    if (config) this.config = { ...this.config, ...config };

    // Terminate existing worker (aborts ongoing search)
    this.initWorker();
    this.stopClock();

    this.chess = new Chess();
    this.gameOver = false;
    this.moveHistory = [];
    this.pendingPromoFrom = null;
    this.pendingPromoTo = null;
    this.engineRunning = false;

    this.players = {
      w: { name: this.config.engineSide === 'w' ? 'Engine' : 'You', timeRemainingMs: this.config.timeControl * 1000 },
      b: { name: this.config.engineSide === 'b' ? 'Engine' : 'You', timeRemainingMs: this.config.timeControl * 1000 },
    };

    this.board.updatePosition(this.chess);
    this.board.setEnabled(this.isHumanTurn());
    this.onClockUpdate?.(this.players.w.timeRemainingMs, this.players.b.timeRemainingMs);
    this.onMoveHistoryUpdate?.(this.moveHistory);
    this.onCapturesUpdate?.([], []);
    this.onEngineThinking?.(false);

    this.startClock();

    if (!this.isHumanTurn()) {
      setTimeout(() => this.runEngine(), 500);
    }
  }

  destroy() {
    this.stopClock();
    this.worker?.terminate();
    this.worker = null;
  }

  flipBoard() { this.board.flip(); }

  undoMove() {
    if (this.engineRunning || this.gameOver) return;
    if (this.config.mode === 'human-engine') {
      this.chess.undo();
      this.chess.undo();
      this.moveHistory.splice(-2);
    } else {
      this.chess.undo();
      this.moveHistory.splice(-1);
    }
    this.board.updatePosition(this.chess);
    this.onMoveHistoryUpdate?.(this.moveHistory);
    this.onScoreUpdate?.(0);
    this.updateCaptures();
  }

  setEngineTime(ms: number) { this.config.engineTimeMs = ms; }
  setEngineMode(mode: 'ab' | 'mcts' | 'nn') { this.engineMode = mode; }
  getEngineMode() { return this.engineMode; }

  // ── Move handling ─────────────────────────────────────────────────────────

  private handleHumanMove(from: Square, to: Square, promotion?: string) {
    if (this.gameOver || this.engineRunning) return;

    const result = this.chess.move({ from, to, promotion: (promotion ?? 'q') as any });
    if (!result) return;

    // Apply increment to the player who just moved (now it's the other turn)
    const movedColor = result.color as 'w' | 'b';
    this.applyIncrement(movedColor);

    this.moveHistory.push({ san: result.san, lan: result.lan, color: movedColor });
    this.board.updatePosition(this.chess, [from, to]);
    this.onMoveHistoryUpdate?.(this.moveHistory);
    this.updateCaptures();

    if (this.checkGameOver()) return;

    if (this.config.mode === 'human-engine') {
      this.board.setEnabled(false);
      // Small delay so the UI can render the human's move before engine starts
      setTimeout(() => this.runEngine(), 80);
    }
  }

  private handlePromotionNeeded(from: Square, to: Square) {
    this.pendingPromoFrom = from;
    this.pendingPromoTo = to;
    this.onPromotionNeeded?.(from, to);
  }

  completePromotion(piece: string) {
    if (!this.pendingPromoFrom || !this.pendingPromoTo) return;
    this.handleHumanMove(this.pendingPromoFrom, this.pendingPromoTo, piece);
    this.pendingPromoFrom = null;
    this.pendingPromoTo = null;
  }

  // ── Engine (runs in Web Worker) ───────────────────────────────────────────

  private async runEngine() {
    if (this.gameOver || this.engineRunning) return;
    this.engineRunning = true;
    this.onEngineThinking?.(true);

    const fen = this.chess.fen();

    try {
      // This runs in a Worker — main thread stays free, clocks keep ticking!
      const result = await this.runEngineInWorker(
        fen,
        this.config.engineTimeMs,
        this.config.engineDepth,
      );

      this.onEngineInfo?.(result);
      this.onScoreUpdate?.(result.score);

      if (result.move && !this.gameOver) {
        const moveResult = this.chess.move(result.move);
        if (moveResult) {
          const from = moveResult.from as Square;
          const to = moveResult.to as Square;
          const movedColor = moveResult.color as 'w' | 'b';
          this.applyIncrement(movedColor);
          this.moveHistory.push({ san: moveResult.san, lan: moveResult.lan, color: movedColor });
          this.board.updatePosition(this.chess, [from, to]);
          this.onMoveHistoryUpdate?.(this.moveHistory);
          this.updateCaptures();
          this.checkGameOver();
        }
      }
    } catch (err) {
      // Worker was terminated (e.g. new game started) — ignore
      console.warn('Engine search cancelled', err);
    }

    this.engineRunning = false;
    this.onEngineThinking?.(false);
    if (!this.gameOver) {
      this.board.setEnabled(this.isHumanTurn());
    }
  }

  // ── Clock ────────────────────────────────────────────────────────────────
  // Only ticks for the player whose turn it currently is.
  // Since engine runs in a Worker, the main thread is free and setInterval fires normally.

  private startClock() {
    this.stopClock();
    let lastTick = Date.now();

    this.clockInterval = setInterval(() => {
      if (this.gameOver) return;

      const now = Date.now();
      const delta = now - lastTick;
      lastTick = now;

      const turn = this.chess.turn() as 'w' | 'b';
      this.players[turn].timeRemainingMs = Math.max(0, this.players[turn].timeRemainingMs - delta);
      this.onClockUpdate?.(this.players.w.timeRemainingMs, this.players.b.timeRemainingMs);

      if (this.players[turn].timeRemainingMs === 0) {
        this.stopClock();
        this.gameOver = true;
        this.board.setEnabled(false);
        const winner = turn === 'w' ? 'Black' : 'White';
        this.onGameOver?.(`${winner} wins`, `${turn === 'w' ? 'White' : 'Black'} ran out of time`);
      }
    }, 100);
  }

  private stopClock() {
    if (this.clockInterval) {
      clearInterval(this.clockInterval);
      this.clockInterval = null;
    }
  }

  private applyIncrement(color: 'w' | 'b') {
    this.players[color].timeRemainingMs += this.config.increment * 1000;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private isHumanTurn(): boolean {
    if (this.config.mode === 'human-human') return true;
    if (this.config.mode === 'engine-engine') return false;
    return this.chess.turn() !== this.config.engineSide;
  }

  private checkGameOver(): boolean {
    let result: string | null = null;
    let reason: string | null = null;

    if (this.chess.isCheckmate()) {
      const winner = this.chess.turn() === 'w' ? 'Black' : 'White';
      result = `${winner} wins`;
      reason = 'by checkmate';
    } else if (this.chess.isStalemate()) {
      result = 'Draw'; reason = 'by stalemate';
    } else if (this.chess.isInsufficientMaterial()) {
      result = 'Draw'; reason = 'by insufficient material';
    } else if (this.chess.isThreefoldRepetition()) {
      result = 'Draw'; reason = 'by threefold repetition';
    } else if (this.chess.isDraw()) {
      result = 'Draw'; reason = 'by 50-move rule';
    }

    if (result && reason) {
      this.gameOver = true;
      this.stopClock();
      this.board.setEnabled(false);
      this.onGameOver?.(result, reason);
      return true;
    }
    return false;
  }

  private updateCaptures() {
    const history = this.chess.history({ verbose: true });
    const wCaptures: string[] = [];
    const bCaptures: string[] = [];
    for (const move of history) {
      if (move.captured) {
        if (move.color === 'w') wCaptures.push(move.captured);
        else bCaptures.push(move.captured);
      }
    }
    this.onCapturesUpdate?.(wCaptures, bCaptures);
  }

  getChess() { return this.chess; }
  getBoard() { return this.board; }
  isGameOver() { return this.gameOver; }
}
