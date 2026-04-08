"""Prompt templates for video analysis."""

SCENE_ANALYSIS = """\
You are analyzing a chess video frame. Describe what you see:

1. Is this an over-the-board (OTB) chess game or a digital/online game?
2. Is there a 2D board overlay visible (a computer-rendered board diagram)?
3. Where is the chess board located in the frame? (e.g., center, left, right)
4. Can you identify any players or their names (from nameplates, graphics)?
5. What phase of the game does this appear to be? (opening, middlegame, endgame)
6. Are there any clocks, score displays, or other relevant information visible?

Respond in JSON format:
{
    "scene_type": "otb" | "online" | "broadcast",
    "has_overlay": true | false,
    "board_location": "description of board position in frame",
    "players": {"white": "name or unknown", "black": "name or unknown"},
    "game_phase": "opening" | "middlegame" | "endgame" | "unknown",
    "time_control": "description or unknown",
    "additional_notes": "any other relevant observations"
}"""

BOARD_READING = """\
You are looking at a chess board. For each square, identify the piece.
Use standard chess notation:
- K=King, Q=Queen, R=Rook, B=Bishop, N=Knight, P=Pawn
- Uppercase = White, lowercase = black
- Empty squares = "."

Read the board from rank 8 (top) to rank 1 (bottom), files a-h (left to right).
Output the position as a FEN piece placement string (ranks separated by "/").
For empty squares, use the count (e.g., "8" for an empty rank).

Respond with just the FEN piece placement, nothing else.
Example: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"""
