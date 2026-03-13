import { Chess } from 'chess.js';
import type { Square } from 'chess.js';
import { getPieceSvgUrl } from '../pieces';
import type { PieceColor, PieceType } from '../pieces';

export type BoardOrientation = 'white' | 'black';

interface DragState {
  fromSq: Square;
  pieceEl: HTMLImageElement;
  startX: number;
  startY: number;
}

export class ChessBoard {
  private container: HTMLElement;
  private chess: Chess;
  private orientation: BoardOrientation = 'white';
  private selected: Square | null = null;
  private legalTargets: Square[] = [];
  private lastMove: [Square, Square] | null = null;
  private drag: DragState | null = null;
  private onMoveCallback: ((from: Square, to: Square, promotion?: string) => void) | null = null;
  private onPromotionNeeded: ((from: Square, to: Square) => void) | null = null;
  private enabled = true;

  constructor(container: HTMLElement, chess: Chess) {
    this.container = container;
    this.chess = chess;
    this.render();
    this.bindDrag();
  }

  setOrientation(o: BoardOrientation) {
    this.orientation = o;
    this.render();
  }

  flip() {
    this.orientation = this.orientation === 'white' ? 'black' : 'white';
    this.render();
  }

  setEnabled(v: boolean) {
    this.enabled = v;
  }

  onMove(cb: (from: Square, to: Square, promotion?: string) => void) {
    this.onMoveCallback = cb;
  }

  onPromotion(cb: (from: Square, to: Square) => void) {
    this.onPromotionNeeded = cb;
  }

  updatePosition(chess: Chess, lastMove?: [Square, Square]) {
    this.chess = chess;
    this.selected = null;
    this.legalTargets = [];
    if (lastMove) this.lastMove = lastMove;
    this.render();
  }

  // ── Rendering ─────────────────────────────────────────────────────────────

  render() {
    this.container.innerHTML = '';
    const board = this.chess.board();

    for (let visualRow = 0; visualRow < 8; visualRow++) {
      for (let visualCol = 0; visualCol < 8; visualCol++) {
        // chess.js: board[0][0] = a8, board[7][7] = h1
        // White view: top row = rank 8 = board row 0
        // Black view: top row = rank 1 = board row 7
        const boardRow = this.orientation === 'white' ? visualRow : 7 - visualRow;
        const boardCol = this.orientation === 'white' ? visualCol : 7 - visualCol;

        const rank = 8 - boardRow;                        // board row 0 = rank 8
        const file = String.fromCharCode(97 + boardCol);  // col 0 = 'a'
        const sq = `${file}${rank}` as Square;

        const isLight = (boardRow + boardCol) % 2 === 1;
        const sqEl = document.createElement('div');
        sqEl.className = `square ${isLight ? 'light' : 'dark'}`;
        sqEl.dataset.square = sq;

        if (sq === this.selected) sqEl.classList.add('selected');
        if (this.lastMove && this.lastMove.includes(sq)) sqEl.classList.add('last-move');

        if (this.chess.inCheck()) {
          const kingSq = this.findKingSquare(this.chess.turn() as 'w' | 'b');
          if (kingSq === sq) sqEl.classList.add('in-check');
        }

        // Labels: rank on left column, file on bottom row
        if (visualCol === 0) {
          const label = document.createElement('span');
          label.className = 'rank-label';
          label.textContent = `${rank}`;
          sqEl.appendChild(label);
        }
        if (visualRow === 7) {
          const label = document.createElement('span');
          label.className = 'file-label';
          label.textContent = file;
          sqEl.appendChild(label);
        }

        // Legal move dot / ring
        if (this.legalTargets.includes(sq)) {
          const piece = board[boardRow][boardCol];
          const indicator = document.createElement('div');
          indicator.className = piece ? 'capture-ring' : 'move-dot';
          sqEl.appendChild(indicator);
        }

        // Piece
        const piece = board[boardRow][boardCol];
        if (piece) {
          const img = document.createElement('img');
          img.className = 'piece';
          img.src = getPieceSvgUrl(piece.color as PieceColor, piece.type as PieceType);
          img.alt = `${piece.color}${piece.type}`;
          img.dataset.square = sq;
          img.draggable = false;
          sqEl.appendChild(img);
        }

        sqEl.addEventListener('click', () => this.handleSquareClick(sq));
        this.container.appendChild(sqEl);
      }
    }
  }

  private findKingSquare(color: 'w' | 'b'): Square | null {
    const board = this.chess.board();
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = board[r][c];
        if (p && p.type === 'k' && p.color === color) {
          const rank = 8 - r;                       // row 0 = rank 8
          const file = String.fromCharCode(97 + c);
          return `${file}${rank}` as Square;
        }
      }
    }
    return null;
  }

  // ── Click handling ────────────────────────────────────────────────────────

  private handleSquareClick(sq: Square) {
    if (!this.enabled) return;

    // If a piece is selected, try to move
    if (this.selected) {
      if (this.legalTargets.includes(sq)) {
        this.attemptMove(this.selected, sq);
        return;
      }
      // Reselect
      this.selected = null;
      this.legalTargets = [];
    }

    // Select piece
    const board = this.chess.board();
    const [r, c] = this.squareToRC(sq);
    const piece = board[r][c];
    if (piece && piece.color === this.chess.turn()) {
      this.selected = sq;
      this.legalTargets = this.chess.moves({ square: sq, verbose: true }).map(m => m.to as Square);
    }

    this.render();
  }

  private attemptMove(from: Square, to: Square) {
    // Check promotion
    const board = this.chess.board();
    const [r, c] = this.squareToRC(from);
    const piece = board[r][c];
    const isPromotion = piece?.type === 'p' && (to[1] === '8' || to[1] === '1');

    if (isPromotion) {
      this.onPromotionNeeded?.(from, to);
    } else {
      this.onMoveCallback?.(from, to);
    }
    this.selected = null;
    this.legalTargets = [];
  }

  private squareToRC(sq: Square): [number, number] {
    const col = sq.charCodeAt(0) - 97;   // a=0..h=7
    const rank = parseInt(sq[1]);          // 1..8
    const row = 8 - rank;                  // chess.js: row 0 = rank 8
    return [row, col];
  }

  // ── Drag handling ────────────────────────────────────────────────────────

  private bindDrag() {
    this.container.addEventListener('mousedown', (e) => this.onMouseDown(e));
    window.addEventListener('mousemove', (e) => this.onMouseMove(e));
    window.addEventListener('mouseup', (e) => this.onMouseUp(e));
  }

  private _didDrag = false;

  private onMouseDown(e: MouseEvent) {
    if (!this.enabled) return;
    const img = (e.target as HTMLElement).closest('img.piece') as HTMLImageElement | null;
    if (!img) return;

    const sq = img.dataset.square as Square;
    const [r, c] = this.squareToRC(sq);
    const piece = this.chess.board()[r][c];
    if (!piece || piece.color !== this.chess.turn()) return;

    this._didDrag = false;

    // Create floating drag ghost (hidden initially until first move)
    const ghost = document.createElement('img');
    ghost.className = 'piece dragging';
    ghost.src = getPieceSvgUrl(piece.color as PieceColor, piece.type as PieceType);
    ghost.style.left = `${e.clientX}px`;
    ghost.style.top = `${e.clientY}px`;
    ghost.style.opacity = '0'; // invisible until drag threshold
    document.body.appendChild(ghost);

    this.drag = { fromSq: sq, pieceEl: ghost, startX: e.clientX, startY: e.clientY };
  }

  private onMouseMove(e: MouseEvent) {
    if (!this.drag) return;
    const dx = e.clientX - this.drag.startX;
    const dy = e.clientY - this.drag.startY;

    if (!this._didDrag && Math.sqrt(dx * dx + dy * dy) > 5) {
      // Threshold crossed — this is a real drag
      this._didDrag = true;
      this.drag.pieceEl.style.opacity = '1';
      // Select piece and show dots
      this.selected = this.drag.fromSq;
      this.legalTargets = this.chess.moves({ square: this.drag.fromSq, verbose: true }).map(m => m.to as Square);
      this.render();
    }

    if (this._didDrag) {
      this.drag.pieceEl.style.left = `${e.clientX}px`;
      this.drag.pieceEl.style.top = `${e.clientY}px`;
    }
  }

  private onMouseUp(e: MouseEvent) {
    if (!this.drag) return;
    const ghost = this.drag.pieceEl;
    document.body.removeChild(ghost);

    if (this._didDrag) {
      // Drag-and-drop: resolve the drop target
      const el = document.elementFromPoint(e.clientX, e.clientY);
      const sqEl = el?.closest('[data-square]') as HTMLElement | null;
      const toSq = sqEl?.dataset.square as Square | undefined;

      if (toSq && this.legalTargets.includes(toSq)) {
        this.attemptMove(this.drag.fromSq, toSq);
      } else {
        this.selected = null;
        this.legalTargets = [];
      }
      this.render();
    }
    // If NOT a drag (pure click), let the click event on the square handle it

    this.drag = null;
    this._didDrag = false;
  }
}
