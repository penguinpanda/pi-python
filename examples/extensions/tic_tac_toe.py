"""Tic-tac-toe tool - stateless board in params/details.

Python port of tic-tac-toe.ts（无共享状态：模型每次传完整 board，避免并行竞态）。
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def _winner(board: list[str]) -> str | None:
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def create_extension(pi: ExtensionAPI):
    def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        board = [str(cell) for cell in params.get("board", [""] * 9)]
        move = params.get("move")
        player = params.get("player", "X")
        if not isinstance(move, int) or not 0 <= move < 9:
            return {
                "content": [{"type": "text", "text": "Invalid move: must be 0-8"}],
                "details": {"board": board, "winner": None, "done": False},
            }
        if board[move]:
            return {
                "content": [{"type": "text", "text": "Cell already taken"}],
                "details": {"board": board, "winner": None, "done": False},
            }
        board[move] = player
        winner = _winner(board)
        done = bool(winner) or all(board)
        status = (
            f"Player {winner} wins!"
            if winner
            else ("Draw" if all(board) else f"{'O' if player == 'X' else 'X'} to move")
        )
        return {
            "content": [{"type": "text", "text": status}],
            "details": {"board": board, "winner": winner, "done": done},
        }

    pi.register_tool(
        ToolDefinition(
            name="tictactoe",
            label="Tic-tac-toe",
            description="Play tic-tac-toe. Pass the full board (9 cells) and a move index.",
            parameters={
                "type": "object",
                "properties": {
                    "board": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "9 cells: 'X', 'O', or ''",
                    },
                    "move": {"type": "number", "description": "Move index 0-8"},
                    "player": {"type": "string", "enum": ["X", "O"]},
                },
                "required": ["board", "move", "player"],
            },
            execute=execute,
        )
    )
