// ── MCTS Node ─────────────────────────────────────────────────────────────────
// Each node represents a chess position reachable by a specific move.
// Stats are maintained in the PUCT style used by AlphaZero / LC0.

export class MCTSNode {
  /** Move (LAN) that led to this position from parent. Null for root. */
  move:       string | null;
  parent:     MCTSNode | null;
  children:   MCTSNode[] | null; // null = not yet expanded

  /** N — visit count */
  visits:     number;
  /** W — total value accumulated from this node (from perspective of node's player) */
  totalValue: number;
  /** P — prior probability from policy network (or 1/N if uniform) */
  prior:      number;

  /** Terminal state flags */
  isTerminal: boolean;
  terminalValue: number; // 1 = current side won, -1 = lost, 0 = draw

  constructor(move: string | null, parent: MCTSNode | null, prior: number) {
    this.move       = move;
    this.parent     = parent;
    this.children   = null;
    this.visits     = 0;
    this.totalValue = 0;
    this.prior      = prior;
    this.isTerminal = false;
    this.terminalValue = 0;
  }

  /** Q — mean value estimate */
  get q(): number {
    return this.visits === 0 ? 0 : this.totalValue / this.visits;
  }

  /**
   * PUCT score: used for child selection.
   * U = C_puct * P * sqrt(N_parent) / (1 + N_child)
   * score = Q + U
   */
  puct(parentVisits: number, cPuct: number): number {
    const u = cPuct * this.prior * Math.sqrt(parentVisits) / (1 + this.visits);
    return this.q + u;
  }

  /** Select child with highest PUCT score */
  selectChild(cPuct: number): MCTSNode {
    let best: MCTSNode | null = null;
    let bestScore = -Infinity;
    for (const child of this.children!) {
      const score = child.puct(this.visits, cPuct);
      if (score > bestScore) { bestScore = score; best = child; }
    }
    return best!;
  }

  /** Most-visited child (best move at root after search) */
  bestChild(): MCTSNode {
    let best: MCTSNode | null = null;
    let bestVisits = -1;
    for (const child of this.children!) {
      if (child.visits > bestVisits) { bestVisits = child.visits; best = child; }
    }
    return best!;
  }

  /** Backpropagate value up the tree */
  backpropagate(value: number): void {
    this.visits++;
    this.totalValue += value;
    if (this.parent) {
      // Flip value: parent sees this from opponent perspective
      this.parent.backpropagate(-value);
    }
  }
}
