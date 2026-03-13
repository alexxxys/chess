import './styles/main.css';
import { GameController } from './game/GameController';
import type { Square } from 'chess.js';
import { getPieceSvgUrl } from './pieces';

// ── DOM Setup ──────────────────────────────────────────────────────────────

const app = document.getElementById('app')!;

app.innerHTML = `
<header class="app-header">
  <div class="app-logo">
    <span class="crown">♛</span>
    <span>DeepChess</span>
  </div>
  <div class="header-controls">
    <button class="btn" id="btn-flip" title="Flip board">⇅ Flip</button>
    <button class="btn" id="btn-new" title="New game">＋ New Game</button>
  </div>
</header>

<div class="main-content">
  <!-- Left Sidebar -->
  <div class="sidebar-left">
    <div class="sidebar-section">
      <div class="section-title">Engine</div>
      <div class="engine-panel">
        <div class="engine-status">
          <div class="engine-dot" id="engine-dot"></div>
          <div class="engine-info">
            <div class="engine-label" id="engine-label">Alpha-Beta Search</div>
            <div class="engine-detail" id="engine-detail">Ready</div>
          </div>
        </div>
        <div class="engine-mode-toggle">
          <button class="mode-btn active" id="mode-ab">⚡ Alpha-Beta</button>
          <button class="mode-btn" id="mode-nn">🧠 MCTS + NN</button>
        </div>
        <div class="score-bar-container">
          <div class="score-text">
            <span>Black</span>
            <span id="score-value">0.00</span>
            <span>White</span>
          </div>
          <div class="score-bar">
            <div class="score-bar-fill" id="score-bar" style="width:50%"></div>
          </div>
        </div>
        <div class="strength-control">
          <div class="strength-header">
            <span>Engine Strength</span>
            <span class="strength-value" id="strength-label">2000ms</span>
          </div>
          <input type="range" id="strength-slider" min="100" max="5000" step="100" value="2000" />
        </div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="section-title">Time Control</div>
      <div class="time-controls">
        <button class="time-btn" data-time="60" data-inc="0">1+0</button>
        <button class="time-btn" data-time="180" data-inc="0">3+0</button>
        <button class="time-btn" data-time="300" data-inc="0">5+0</button>
        <button class="time-btn active" data-time="600" data-inc="0">10+0</button>
        <button class="time-btn" data-time="900" data-inc="10">15+10</button>
        <button class="time-btn" data-time="1800" data-inc="0">30+0</button>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="section-title">Play As</div>
      <div class="btn-row">
        <button class="btn active" id="play-white" style="border-color: var(--accent);">♔ White</button>
        <button class="btn" id="play-black">♚ Black</button>
      </div>
    </div>

    <div class="sidebar-section" style="margin-top:auto; border-top: 1px solid var(--border); border-bottom:none;">
      <div class="game-controls">
        <div class="btn-row">
          <button class="btn" id="btn-undo">← Undo</button>
          <button class="btn btn-danger" id="btn-resign">Resign</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Board area -->
  <div class="board-area">
    <div class="player-bar" id="top-player-bar">
      <div class="player-info">
        <div class="player-avatar black-avatar" id="top-avatar">E</div>
        <div>
          <div class="player-name" id="top-name">Engine</div>
          <div class="player-captures" id="top-captures"></div>
        </div>
      </div>
      <div class="player-clock" id="top-clock">10:00</div>
    </div>

    <div class="board-wrapper">
      <div class="board-container" id="board"></div>
    </div>

    <div class="player-bar" id="bottom-player-bar">
      <div class="player-info">
        <div class="player-avatar white-avatar" id="bottom-avatar">Y</div>
        <div>
          <div class="player-name" id="bottom-name">You</div>
          <div class="player-captures" id="bottom-captures"></div>
        </div>
      </div>
      <div class="player-clock active" id="bottom-clock">10:00</div>
    </div>
  </div>

  <!-- Right Sidebar -->
  <div class="sidebar-right">
    <div class="move-history-section">
      <div class="section-title">Move History</div>
      <div class="move-history-wrapper">
        <table class="move-table" id="move-table">
          <tbody id="move-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- Game Over Modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <span class="modal-icon" id="modal-icon">♟</span>
    <div class="modal-title" id="modal-title">Game Over</div>
    <div class="modal-subtitle" id="modal-subtitle"></div>
    <div class="modal-actions">
      <button class="btn btn-primary" id="modal-new">New Game</button>
      <button class="btn" id="modal-close">Close</button>
    </div>
  </div>
</div>

<!-- Promotion Picker -->
<div class="promo-overlay" id="promo-overlay">
  <div class="promo-picker" id="promo-picker">
    <button class="promo-btn" data-piece="q"><img src="" alt="Queen" id="promo-q"/></button>
    <button class="promo-btn" data-piece="r"><img src="" alt="Rook" id="promo-r"/></button>
    <button class="promo-btn" data-piece="b"><img src="" alt="Bishop" id="promo-b"/></button>
    <button class="promo-btn" data-piece="n"><img src="" alt="Knight" id="promo-n"/></button>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>
`;

// ── State ────────────────────────────────────────────────────────────────────

let selectedTime = 600;
let selectedInc = 0;
let playerSide: 'w' | 'b' = 'w';

// ── Controller ────────────────────────────────────────────────────────────────

let ctrl = createController();

function createController() {
  const boardEl = document.getElementById('board')!;
  boardEl.innerHTML = '';

  const gc = new GameController(boardEl, {
    mode: 'human-engine',
    engineSide: playerSide === 'w' ? 'b' : 'w',
    timeControl: selectedTime,
    increment: selectedInc,
    engineDepth: 20,
    engineTimeMs: parseInt((document.getElementById('strength-slider') as HTMLInputElement)?.value ?? '2000'),
  });

  gc.onClockUpdate = (wMs, bMs) => {
    const topIsBlack = playerSide === 'w';
    setClockDisplay('top-clock', topIsBlack ? bMs : wMs);
    setClockDisplay('bottom-clock', topIsBlack ? wMs : bMs);

    // Active clock = whoever's turn it is right now
    const turn = gc.getChess().turn();
    const topActive = topIsBlack ? turn === 'b' : turn === 'w';
    document.getElementById('top-clock')!.classList.toggle('active', topActive);
    document.getElementById('bottom-clock')!.classList.toggle('active', !topActive);
    document.getElementById('top-clock')!.classList.toggle('low-time', topIsBlack ? bMs < 30000 : wMs < 30000);
    document.getElementById('bottom-clock')!.classList.toggle('low-time', topIsBlack ? wMs < 30000 : bMs < 30000);
  };

  gc.onEngineThinking = (thinking) => {
    const dot = document.getElementById('engine-dot')!;
    const detail = document.getElementById('engine-detail')!;
    dot.className = thinking ? 'engine-dot thinking' : 'engine-dot ready';
    if (thinking) detail.textContent = 'Thinking...';
  };

  gc.onEngineInfo = (info) => {
    const dot = document.getElementById('engine-dot')!;
    const detail = document.getElementById('engine-detail')!;
    dot.className = 'engine-dot ready';
    const scoreStr = Math.abs(info.score) > 900000 ? 'M' : (info.score / 100).toFixed(2);
    detail.textContent = `d${info.depth} | ${scoreStr}cp | ${info.nodes.toLocaleString()} nodes | ${info.thinkMs}ms`;
  };

  gc.onMoveHistoryUpdate = (moves) => {
    renderMoveHistory(moves);
  };

  gc.onGameOver = (result, reason) => {
    showGameOver(result, reason);
  };

  gc.onPromotionNeeded = (from, to) => {
    showPromotion(from, to, gc);
  };

  gc.onScoreUpdate = (score) => {
    updateScoreBar(score, gc);
  };

  gc.onCapturesUpdate = (wCap, bCap) => {
    renderCaptures(wCap, bCap);
  };

  // Initial state
  document.getElementById('engine-dot')!.className = 'engine-dot ready';

  return gc;
}

// ── Clock display ──────────────────────────────────────────────────────────

function setClockDisplay(id: string, ms: number) {
  const el = document.getElementById(id)!;
  const totalSec = Math.ceil(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  el.textContent = `${min}:${sec.toString().padStart(2, '0')}`;
}

// ── Score bar ──────────────────────────────────────────────────────────────

function updateScoreBar(rawScore: number, gc?: GameController) {
  // Score is from engine's perspective (side to move). Convert to white-relative.
  const chess = (gc ?? ctrl).getChess();
  const whiteScore = chess.turn() === 'w' ? rawScore : -rawScore;
  const pct = 50 + Math.max(-50, Math.min(50, whiteScore / 20));
  document.getElementById('score-bar')!.style.width = `${pct}%`;

  const scoreVal = document.getElementById('score-value')!;
  if (Math.abs(whiteScore) > 900000) {
    scoreVal.textContent = whiteScore > 0 ? '+M' : '-M';
  } else {
    const cp = (whiteScore / 100).toFixed(2);
    scoreVal.textContent = whiteScore > 0 ? `+${cp}` : cp;
  }
}

// ── Move history ───────────────────────────────────────────────────────────

function renderMoveHistory(moves: { san: string; color: 'w' | 'b' }[]) {
  const tbody = document.getElementById('move-tbody')!;
  tbody.innerHTML = '';

  for (let i = 0; i < moves.length; i += 2) {
    const tr = document.createElement('tr');
    const num = Math.floor(i / 2) + 1;

    const tdNum = document.createElement('td');
    tdNum.className = 'move-num';
    tdNum.textContent = `${num}.`;

    const tdW = document.createElement('td');
    tdW.className = 'move-cell';
    tdW.textContent = moves[i]?.san ?? '';
    if (i === moves.length - 1) tdW.classList.add('current-move');

    const tdB = document.createElement('td');
    tdB.className = 'move-cell';
    tdB.textContent = moves[i + 1]?.san ?? '';
    if (i + 1 === moves.length - 1) tdB.classList.add('current-move');

    tr.append(tdNum, tdW, tdB);
    tbody.appendChild(tr);
  }

  // Scroll to bottom
  const wrapper = document.querySelector('.move-history-wrapper')!;
  wrapper.scrollTop = wrapper.scrollHeight;
}

// ── Captures ───────────────────────────────────────────────────────────────

const PIECE_EMOJI: Record<string, string> = {
  p: '♟', n: '♞', b: '♝', r: '♜', q: '♛',
};
const PIECE_VAL: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9 };

function renderCaptures(wCap: string[], bCap: string[]) {
  const wAdv = wCap.reduce((s, p) => s + (PIECE_VAL[p] ?? 0), 0)
             - bCap.reduce((s, p) => s + (PIECE_VAL[p] ?? 0), 0);

  const topIsBlack = playerSide === 'w';
  const topCapEl = document.getElementById('top-captures')!;
  const botCapEl = document.getElementById('bottom-captures')!;

  // Top player (black) sees white's captures as black's material that was taken
  const topPieces = topIsBlack ? wCap : bCap;
  const botPieces = topIsBlack ? bCap : wCap;
  const topAdv = topIsBlack ? -wAdv : wAdv;

  topCapEl.innerHTML = topPieces.map(p => `<span class="capture-piece">${PIECE_EMOJI[p] ?? ''}</span>`).join('')
    + (topAdv > 0 ? `<span class="material-advantage">+${topAdv}</span>` : '');

  botCapEl.innerHTML = botPieces.map(p => `<span class="capture-piece">${PIECE_EMOJI[p] ?? ''}</span>`).join('')
    + (topAdv < 0 ? `<span class="material-advantage">+${Math.abs(topAdv)}</span>` : '');
}

// ── Game Over Modal ────────────────────────────────────────────────────────

function showGameOver(result: string, reason: string) {
  const overlay = document.getElementById('modal-overlay')!;
  const icon = document.getElementById('modal-icon')!;
  const title = document.getElementById('modal-title')!;
  const subtitle = document.getElementById('modal-subtitle')!;

  icon.textContent = result.includes('Draw') ? '🤝' : result.includes('White') ? '♔' : '♚';
  title.textContent = result;
  subtitle.textContent = reason;
  overlay.classList.add('visible');
}

document.getElementById('modal-new')!.addEventListener('click', () => {
  document.getElementById('modal-overlay')!.classList.remove('visible');
  ctrl.newGame();
});

document.getElementById('modal-close')!.addEventListener('click', () => {
  document.getElementById('modal-overlay')!.classList.remove('visible');
});

// ── Promotion Picker ───────────────────────────────────────────────────────

function showPromotion(_from: Square, _to: Square, gc: GameController) {
  const movingColor = gc.getChess().turn() as 'w' | 'b';
  const pieces: Array<{ piece: 'q'|'r'|'b'|'n', id: string }> = [
    { piece: 'q', id: 'promo-q' }, { piece: 'r', id: 'promo-r' },
    { piece: 'b', id: 'promo-b' }, { piece: 'n', id: 'promo-n' },
  ];

  for (const { piece, id } of pieces) {
    const img = document.getElementById(id) as HTMLImageElement;
    img.src = getPieceSvgUrl(movingColor, piece);
  }

  const overlay = document.getElementById('promo-overlay')!;
  overlay.classList.add('visible');

  const picker = document.getElementById('promo-picker')!;
  const handler = (e: Event) => {
    const btn = (e.target as HTMLElement).closest('[data-piece]') as HTMLElement | null;
    if (!btn) return;
    overlay.classList.remove('visible');
    picker.removeEventListener('click', handler);
    gc.completePromotion(btn.dataset.piece!);
  };
  picker.addEventListener('click', handler);
}

// ── Strength slider ────────────────────────────────────────────────────────

const slider = document.getElementById('strength-slider') as HTMLInputElement;
const strengthLabel = document.getElementById('strength-label')!;

slider.addEventListener('input', () => {
  const ms = parseInt(slider.value);
  strengthLabel.textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
  ctrl.setEngineTime(ms);
});

// ── Engine mode toggle ──────────────────────────────────────────────────

function applyEngineMode(mode: 'ab' | 'nn') {
  ctrl.setEngineMode(mode === 'nn' ? 'nn' : 'ab');
  const label = document.getElementById('engine-label')!;
  const btnAb = document.getElementById('mode-ab')!;
  const btnNn = document.getElementById('mode-nn')!;
  if (mode === 'nn') {
    label.textContent = '🧠 MCTS + Neural Network';
    (label as HTMLElement).style.color = 'var(--accent)';
    btnNn.classList.add('active');
    btnAb.classList.remove('active');
  } else {
    label.textContent = '⚡ Alpha-Beta Search';
    (label as HTMLElement).style.color = '';
    btnAb.classList.add('active');
    btnNn.classList.remove('active');
  }
}

document.getElementById('mode-ab')!.addEventListener('click', () => {
  applyEngineMode('ab');
  showToast('Switched to Alpha-Beta Search');
});
document.getElementById('mode-nn')!.addEventListener('click', () => {
  applyEngineMode('nn');
  showToast('🧠 Switched to MCTS + Neural Network!');
});

// ── Time controls ──────────────────────────────────────────────────

document.querySelectorAll('.time-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedTime = parseInt((btn as HTMLElement).dataset.time!);
    selectedInc = parseInt((btn as HTMLElement).dataset.inc!);
  });
});

// ── Play as ───────────────────────────────────────────────────────────────

document.getElementById('play-white')!.addEventListener('click', () => {
  playerSide = 'w';
  document.getElementById('play-white')!.style.borderColor = 'var(--accent)';
  document.getElementById('play-black')!.style.borderColor = '';
});

document.getElementById('play-black')!.addEventListener('click', () => {
  playerSide = 'b';
  document.getElementById('play-black')!.style.borderColor = 'var(--accent)';
  document.getElementById('play-white')!.style.borderColor = '';
});

// ── Header buttons ─────────────────────────────────────────────────────────

document.getElementById('btn-flip')!.addEventListener('click', () => ctrl.flipBoard());

document.getElementById('btn-new')!.addEventListener('click', () => {
  document.getElementById('modal-overlay')!.classList.remove('visible');
  ctrl.destroy(); // terminate running worker + stop clocks
  ctrl = createController();
  showToast('New game started!');
});

document.getElementById('btn-undo')!.addEventListener('click', () => {
  ctrl.undoMove();
  showToast('Move undone');
});

document.getElementById('btn-resign')!.addEventListener('click', () => {
  const chess = ctrl.getChess();
  const loser = chess.turn() === 'w' ? 'White' : 'Black';
  showGameOver(`${loser === 'White' ? 'Black' : 'White'} wins`, `${loser} resigned`);
});

// ── Toast ──────────────────────────────────────────────────────────────────

function showToast(msg: string) {
  const toast = document.getElementById('toast')!;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2000);
}
