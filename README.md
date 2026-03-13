# ♛ DeepChess

A web-based chess application with a custom chess engine featuring two play modes:
- **⚡ Alpha-Beta Search** — classical minimax with alpha-beta pruning
- **🧠 MCTS + Neural Network** — Monte Carlo Tree Search guided by an AlphaZero-style ResNet trained through self-play

> The neural network runs entirely in the browser via [ONNX Runtime Web](https://onnxruntime.ai/) — no backend server needed.

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | TypeScript + Vite |
| Chess logic | [chess.js](https://github.com/jhlywa/chess.js) |
| Neural network inference | ONNX Runtime Web |
| Training | Python 3.10+, PyTorch 2.x, CUDA |

---

## 🚀 Quick Start (Web App)

### Prerequisites
- [Node.js](https://nodejs.org/) 18+

### Install & Run

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production

```bash
npm run build
npm run preview
```

---

## 🎮 How to Play

### Interface Overview

```
┌──────────────────────────────────────────────────────┐
│  ♛ DeepChess                     ⇅ Flip  ＋ New Game │
├─────────────────┬───────────────┬────────────────────┤
│  Engine Panel   │               │  Move History      │
│  Time Control   │   Chess Board │                    │
│  Play As        │               │                    │
│  Undo / Resign  │               │                    │
└─────────────────┴───────────────┴────────────────────┘
```

### Making Moves

1. **Click** a piece to select it — available squares will be highlighted
2. **Click** a destination square to move
3. When a pawn reaches the last rank, a **promotion picker** appears — choose Queen, Rook, Bishop, or Knight

### Controls

| Control | Description |
|---|---|
| **⇅ Flip** | Rotate the board 180° |
| **＋ New Game** | Start a fresh game instantly |
| **← Undo** | Take back your last move (and the engine's response) |
| **Resign** | Concede the current game |

---

## ⚙️ Settings

All settings are in the **left sidebar** and apply to the next game (or immediately where noted).

### Engine Mode

| Mode | Description |
|---|---|
| **⚡ Alpha-Beta** | Classical search, fast and reliable. Depth 20. |
| **🧠 MCTS + NN** | Neural network guided MCTS. Stronger positional play. |

> Switching mode takes effect **immediately** mid-game.

### Engine Strength

Use the **Engine Strength** slider to adjust thinking time:
- Range: `100ms` → `5000ms`
- Higher = stronger play (more search time)

### Time Control

Choose a chess clock format before starting a new game:

| Format | Description |
|---|---|
| 1+0 | 1 minute, no increment |
| 3+0 | 3 minutes |
| 5+0 | 5 minutes |
| **10+0** | 10 minutes *(default)* |
| 15+10 | 15 minutes + 10s increment |
| 30+0 | 30 minutes |

> ⚠️ Time control can only be changed **before** starting a new game.

### Play As

Choose whether you play as **White** (♔) or **Black** (♚). Takes effect on the next new game.

---

## 🧠 Neural Network Architecture

The engine uses an **AlphaZero-style ResNet**:

```
Input:  [19, 8, 8] — 12 piece planes + castling/en-passant/side-to-move
         ↓
Stem:   Conv(19→128, 3×3) → BatchNorm → ReLU
         ↓
Body:   10 × Residual Blocks (128 channels)
         ↓
     ┌───────────────────┐
     │    Policy Head    │   → [4672] move logits
     ├───────────────────┤
     │    Value Head     │   → scalar ∈ [-1, +1]
     └───────────────────┘
```

| Size preset | Blocks | Channels | Parameters |
|---|---|---|---|
| `tiny` | 4 | 64 | ~500K |
| `small` | 6 | 96 | ~2M |
| **`medium`** | **10** | **128** | **~8M** ← default |
| `large` | 20 | 256 | ~40M |

---

## 🏋️ Training the Neural Network

The training pipeline uses AlphaZero's approach: **self-play → train → evaluate → repeat**.

### Prerequisites

```bash
# Create a Python virtual environment
python -m venv train_env
train_env\Scripts\activate   # Windows
# source train_env/bin/activate  # Linux/macOS

# Install dependencies (CUDA 12.4 recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install chess
```

### Run Training

```bash
# Basic training (200 generations, medium model, auto-detects GPU)
python -m train.pipeline

# Custom settings
python -m train.pipeline \
  --gens 200 \
  --games 100 \
  --sims 200 \
  --size medium \
  --device cuda

# Start fresh (ignore existing checkpoint)
python -m train.pipeline --fresh
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--gens` | 200 | Number of training generations |
| `--games` | 100 | Self-play games per generation |
| `--sims` | 200 | MCTS simulations per move |
| `--size` | `medium` | Model size: `tiny`, `small`, `medium`, `large` |
| `--device` | auto | `cuda` or `cpu` |
| `--lr` | 1e-3 | Learning rate |
| `--batch` | 512 | Mini-batch size |
| `--replay-size` | 500000 | Replay buffer max size |
| `--eval-every` | 10 | Evaluate new vs old model every N gens |
| `--export-every` | 5 | Export ONNX to `public/` every N gens |
| `--checkpoint-dir` | `checkpoints/` | Directory for `.pt` checkpoints |
| `--fresh` | false | Ignore existing checkpoint, start from scratch |

### Training Loop

Each generation:
1. **Self-Play** — the current model plays against itself using MCTS
2. **Train** — network trained on examples from the replay buffer (policy + value loss)
3. **Evaluate** — new model vs previous checkpoint (promotes if win rate > 55%)
4. **Export ONNX** — model exported to `public/model.onnx` for use in the browser

### Using a New Model in the Browser

After training, the ONNX model is automatically saved to `public/model.onnx`.  
Rebuild the frontend to load it:

```bash
npm run build
npm run preview
```

---

## 📁 Project Structure

```
chess/
├── src/
│   ├── engine/
│   │   ├── core/          # Chess engine core (move generation, zobrist, tables)
│   │   ├── mcts/          # MCTS implementation (TypeScript, browser-side)
│   │   ├── nn/            # ONNX Runtime neural network wrapper
│   │   ├── search.ts      # Alpha-beta search
│   │   ├── eval.ts        # Position evaluation
│   │   ├── openingBook.ts # Opening book
│   │   └── worker.ts      # Web Worker (engine runs off main thread)
│   ├── board/             # Chessboard rendering
│   ├── game/              # Game controller (clocks, move history, promotions)
│   ├── main.ts            # App entry point + UI wiring
│   └── styles/            # CSS
├── train/
│   ├── model.py           # ChessResNet architecture
│   ├── selfplay.py        # Self-play game generation
│   ├── mcts_python.py     # MCTS (Python, for training)
│   ├── input.py           # Board → tensor encoding
│   ├── evaluate.py        # Model evaluation
│   ├── train.py           # Training utilities
│   └── pipeline.py        # Main training pipeline CLI
├── public/
│   └── model.onnx         # Trained model (used by the browser)
├── index.html
├── vite.config.ts
└── package.json
```
