import html
from pathlib import Path

from utils.eval_utils import COLOR_PALETTE, IGNORE_INDEX


def _build_color_class_styles():
    """Generate CSS classes for palette colors to avoid inline styles."""
    css_chunks = []
    for idx, color in enumerate(COLOR_PALETTE):
        css_chunks.append(f".c{idx}{{background-color:{color};}}")
    return "".join(css_chunks)

def _grid_to_html_table(grid, title, allow_html_title=False):
    safe_title = title if allow_html_title else html.escape(title)
    if not grid:
        return (
            f"<div class='grid'><h4>{safe_title}</h4>"
            "<p class='empty'>No data</p></div>"
        )

    rows = []
    for row in grid:
        cells = []
        for value in row:
            try:
                color_index = int(value)
            except (TypeError, ValueError):
                color_index = IGNORE_INDEX

            if not 0 <= color_index < len(COLOR_PALETTE) - 1:
                color_index = len(COLOR_PALETTE) - 1

            cells.append(f"<td class='c{color_index}'></td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = "".join(rows)
    return (
        f"<div class='grid'><h4>{safe_title}</h4>"
        f"<table class='grid-table'>{table_html}</table></div>"
    )


def render_results_html(tasks_payload, metrics, html_path: Path):
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    color_class_styles = _build_color_class_styles()

    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<title>ARC Analysis Results</title>",
        "<style>",
        ":root { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif; color-scheme: light; }",
        "body { margin: 0; background: #f1f4fb; color: #121522; min-height: 100vh; }",
        ".page { max-width: 1200px; margin: 0 auto; padding: 32px 24px 64px; }",
        "h1 { margin: 0 0 8px; font-size: 32px; font-weight: 600; color: #0c1220; }",
        ".metrics { margin-bottom: 32px; }",
        ".metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }",
        ".metric-card { padding: 16px; border-radius: 12px; background: #fff; border: 1px solid #e2e8f0; box-shadow: 0 8px 24px rgba(14,30,64,0.08); }",
        ".metric-label { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: #657098; }",
        ".metric-value { font-size: 24px; font-weight: 600; margin-top: 8px; color: #111728; }",
        ".filters { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; align-items: center; }",
        ".filter-label { font-size: 14px; text-transform: uppercase; letter-spacing: 0.06em; color: #5c647b; }",
        ".filter-button { padding: 8px 16px; border-radius: 999px; border: 1px solid #d4dbeb; background: #fff; color: #111728; cursor: pointer; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; box-shadow: 0 2px 8px rgba(17,23,40,0.08); transition: border 0.2s ease, box-shadow 0.2s ease, color 0.2s ease; }",
        ".filter-button.active { border-color: #2d5bff; color: #2d5bff; box-shadow: 0 6px 18px rgba(45,91,255,0.25); }",
        ".task { margin-bottom: 36px; padding: 24px; border-radius: 16px; background: #fff; border: 1px solid #dde3f3; box-shadow: 0 20px 60px rgba(15,23,43,0.08); }",
        ".task h2 { margin: 0 0 16px; font-size: 20px; font-weight: 600; color: #111728; }",
        ".example { border: 1px solid #e4e8f5; padding: 20px; margin-bottom: 20px; border-radius: 12px; background: #f8faff; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.6); }",
        ".example:last-child { margin-bottom: 0; }",
        ".example h3 { margin: 0 0 16px; font-size: 16px; font-weight: 500; color: #1d2436; }",
        ".grids { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 18px; align-items: start; }",
        ".grid { text-align: center; }",
        ".grid h4 { margin: 8px 0 10px; font-size: 13px; letter-spacing: 0.04em; color: #6a7395; text-transform: uppercase; }",
        ".grid-table { border-collapse: collapse; margin: 0 auto; background: #fff; box-shadow: 0 6px 18px rgba(17,23,40,0.08); }",
        ".grid-table td { width: 8px; height: 8px; border: 1px solid #ffffff; }",
        ".match-label { font-weight: 600; }",
        ".match-correct { color: #15803d; }",
        ".match-false { color: #dc2626; }",
        color_class_styles,
        ".toggle-train { display: inline-flex; align-items: center; gap: 8px; margin: 12px 0; padding: 10px 16px; background: #fff; color: #1a2351; border: 1px solid #ccd4eb; border-radius: 999px; cursor: pointer; letter-spacing: 0.04em; text-transform: uppercase; font-size: 12px; font-weight: 500; transition: box-shadow 0.2s ease, border 0.2s ease; box-shadow: 0 2px 10px rgba(16,24,40,0.08); }",
        ".toggle-train:hover { border-color: #2d5bff; box-shadow: 0 4px 16px rgba(45,91,255,0.25); }",
        ".train-examples { margin: 16px 0 24px; display: flex; gap: 16px; overflow-x: auto; padding-bottom: 8px; }",
        ".train-examples.hidden { display: none; }",
        ".train-example { border: 1px solid #e1e5f3; padding: 20px; border-radius: 12px; background: #fff; flex: 0 0 auto; min-width: 260px; box-shadow: 0 8px 20px rgba(15,23,42,0.08); }",
        ".train-example h3 { margin: 0 0 16px; font-size: 15px; letter-spacing: 0.05em; color: #4d5675; text-transform: uppercase; }",
        ".grids.demo-line { display: flex; flex-wrap: nowrap; gap: 16px; }",
        ".empty { color: #6a7395; font-style: italic; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='page'>",
        "<h1>ARC Analysis Results</h1>",
        "<div class='metrics'>",
        "<div class='metric-grid'>",
        f"<div class='metric-card'><div class='metric-label'>Pass@1</div><div class='metric-value'>{metrics.get('pass_at_1', 0.0):.4f}</div></div>",
        f"<div class='metric-card'><div class='metric-label'>Pass@2</div><div class='metric-value'>{metrics.get('pass_at_2', 0.0):.4f}</div></div>",
        f"<div class='metric-card'><div class='metric-label'>Oracle</div><div class='metric-value'>{metrics.get('oracle', 0.0):.4f}</div></div>",
        f"<div class='metric-card'><div class='metric-label'>Total Tasks</div><div class='metric-value'>{metrics.get('total_tasks', 0)}</div></div>",
        "</div>",
        "</div>",
    ]

    sorted_tasks = sorted(tasks_payload.keys())
    if sorted_tasks:
        lines.append("<div class='filters'>")
        lines.append("<span class='filter-label'>Filter tasks:</span>")
        lines.append("<button class='filter-button active' data-filter='all'>All</button>")
        lines.append("<button class='filter-button' data-filter='correct'>Correct only</button>")
        lines.append("<button class='filter-button' data-filter='failed'>Failed only</button>")
        lines.append("</div>")

    for idx, task_name in enumerate(sorted_tasks, start=1):
        safe_task_name = html.escape(task_name)
        task_payload = tasks_payload[task_name]
        examples = task_payload.get("examples", {})
        train_examples = task_payload.get("train_examples", [])
        task_section_id = f"task-{idx}"
        safe_section_id = html.escape(task_section_id)
        train_container_id = f"train-{task_section_id}"
        escaped_container_id = html.escape(train_container_id)
        ordered_examples = []
        task_has_examples = False
        all_examples_top1 = True
        any_example_top2 = False
        for example_id in sorted(examples.keys(), key=lambda x: int(x)):
            example_data = examples[example_id]
            majority_results = example_data.get("majority_vote") or []
            top1_match = bool(
                majority_results and majority_results[0].get("matches_answer")
            )
            top2_match = False
            if len(majority_results) > 1:
                top2_match = bool(majority_results[1].get("matches_answer"))
            pass_top2 = top1_match or top2_match
            ordered_examples.append((example_id, example_data, majority_results))
            task_has_examples = True
            if not top1_match:
                all_examples_top1 = False
            if pass_top2:
                any_example_top2 = True
        if not task_has_examples:
            task_result = "unknown"
        elif any_example_top2:
            task_result = "correct"
        elif not any_example_top2:
            task_result = "failed"
        else:
            task_result = "partial"
        lines.append(
            f"<section class='task' data-result='{task_result}' id='{safe_section_id}'><h2>Task: {safe_task_name}</h2>"
        )
        if train_examples:
            lines.append(
                f"<button class='toggle-train' data-target='{escaped_container_id}'>Visualize demonstration examples</button>"
            )
            lines.append(f"<div id='{escaped_container_id}' class='train-examples hidden'>")
            for demo_idx, train_example in enumerate(train_examples, start=1):
                safe_idx = html.escape(str(demo_idx))
                lines.append(f"<div class='train-example'><h3>Train Example {safe_idx}</h3>")
                lines.append("<div class='grids demo-line'>")
                lines.append(_grid_to_html_table(train_example.get("input"), "Input"))
                lines.append(_grid_to_html_table(train_example.get("output"), "Output"))
                lines.append("</div></div>")
            lines.append("</div>")
        for example_id, example_data, majority_results in ordered_examples:
            safe_example = html.escape(example_id)
            lines.append(f"<div class='example'><h3>Example {safe_example}</h3>")
            lines.append("<div class='grids'>")
            lines.append(_grid_to_html_table(example_data.get("input"), "Input"))
            lines.append(_grid_to_html_table(example_data.get("answer"), "Ground Truth"))
            if majority_results:
                for rank_idx, entry in enumerate(majority_results[:2], start=1):
                    is_correct = bool(entry.get("matches_answer"))
                    match_text = "Correct" if is_correct else "Incorrect"
                    match_class = "match-correct" if is_correct else "match-false"
                    title = (
                        f"Top {rank_idx} — votes: {entry.get('votes', 0)} — "
                        f"<span class='match-label {match_class}'>{match_text}</span>"
                    )
                    lines.append(
                        _grid_to_html_table(
                            entry.get("prediction"), title, allow_html_title=True
                        )
                    )
            else:
                lines.append("<p class='empty'>No predictions available.</p>")
            lines.append("</div></div>")
        lines.append("</section>")

    lines.extend(
        [
            "<script>",
            "document.addEventListener('DOMContentLoaded', function() {",
            "  document.querySelectorAll('.toggle-train').forEach(function(button) {",
            "    button.addEventListener('click', function() {",
            "      var targetId = button.getAttribute('data-target');",
            "      var container = document.getElementById(targetId);",
            "      if (!container) {",
            "        return;",
            "      }",
            "      container.classList.toggle('hidden');",
            "      var shouldShow = !container.classList.contains('hidden');",
            "      button.textContent = shouldShow ? 'Hide demonstration examples' : 'Visualize demonstration examples';",
            "    });",
            "  });",
            "  var sections = Array.prototype.slice.call(document.querySelectorAll('.task'));",
            "  var filterButtons = Array.prototype.slice.call(document.querySelectorAll('.filter-button'));",
            "  var activeFilter = 'all';",
            "  function applyFilter() {",
            "    sections.forEach(function(section) {",
            "      var result = section.getAttribute('data-result') || 'unknown';",
            "      var shouldShow = (activeFilter === 'all') ||",
            "        (activeFilter === 'correct' && result === 'correct') ||",
            "        (activeFilter === 'failed' && result === 'failed');",
            "      section.style.display = shouldShow ? '' : 'none';",
            "    });",
            "  }",
            "  applyFilter();",
            "  filterButtons.forEach(function(button) {",
            "    button.addEventListener('click', function() {",
            "      var mode = button.getAttribute('data-filter');",
            "      if (mode === activeFilter) {",
            "        return;",
            "      }",
            "      activeFilter = mode;",
            "      filterButtons.forEach(function(btn) { btn.classList.remove('active'); });",
            "      button.classList.add('active');",
            "      applyFilter();",
            "    });",
            "  });",
            "});",
            "</script>",
            "</div>",
            "</body>",
            "</html>",
        ]
    )
    html_content = "\n".join(lines)
    with html_path.open("w", encoding="utf-8") as html_file:
        html_file.write(html_content)
