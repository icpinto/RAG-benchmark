import ast
from pathlib import Path


def test_results_explorer_status_card_does_not_bitwise_and_strings():
    tree = ast.parse(Path("services/ui/app.py").read_text(encoding="utf-8"))
    string_bitwise_and_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitAnd)
        and isinstance(node.left, (ast.Constant, ast.JoinedStr))
        and isinstance(node.right, (ast.Constant, ast.JoinedStr))
    ]

    assert string_bitwise_and_nodes == []
