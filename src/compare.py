"""Bölüm 6 — reports/results/*.json dosyalarından karşılaştırma tablolarını (markdown + LaTeX) üretir."""
import json

import pandas as pd

from common import RESULTS

rows, per_type = [], {}
for f in sorted(RESULTS.glob("*.json")):
    r = json.loads(f.read_text(encoding="utf-8"))
    rows.append({"model": r["model"], "P": r["precision"], "R": r["recall"], "F1": r["f1"],
                 "macro-F1": r["macro_f1"], "R seen": r["recall_by_seen"]["seen"],
                 "R unseen": r["recall_by_seen"]["unseen"]})
    per_type[r["model"]] = {t: v["f1-score"] for t, v in r["per_type"].items() if "avg" not in t}
    per_type[r["model"]]["support"] = None

main = pd.DataFrame(rows).set_index("model")
types = pd.DataFrame(per_type).drop(index="support")
any_r = json.loads(next(RESULTS.glob("*.json")).read_text(encoding="utf-8"))["per_type"]
types["support"] = [any_r[t]["support"] for t in types.index]
types = types.sort_values("support", ascending=False).round(3)

print(main.to_markdown(), "\n")
print(types.to_markdown())
(RESULTS / "main_table.tex").write_text(main.to_latex(float_format="%.3f"), encoding="utf-8")
(RESULTS / "per_type_table.tex").write_text(types.to_latex(float_format="%.3f"), encoding="utf-8")
