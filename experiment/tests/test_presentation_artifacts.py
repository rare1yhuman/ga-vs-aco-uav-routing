import hashlib
import json
import re
import unittest
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscript"
SITE = ROOT / "paper_explainer"
ANALYSIS = ROOT / "experiment" / "results" / "analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_window_json(path: Path, variable: str):
    source = path.read_text()
    payload = re.sub(
        rf"^window\.{re.escape(variable)}\s*=\s*", "", source
    ).rstrip(";\n")
    return json.loads(payload)


class PresentationArtifactTests(unittest.TestCase):
    def test_docx_is_two_column_and_contains_five_figures(self):
        path = MANUSCRIPT / "manuscript.docx"
        self.assertTrue(path.exists())
        with zipfile.ZipFile(path) as archive:
            document = archive.read("word/document.xml").decode("utf-8")
            media = [
                name for name in archive.namelist() if name.startswith("word/media/")
            ]
        self.assertIn("w:num=\"2\"", document)
        self.assertIn("Scalable Memetic Genetic Algorithm", document)
        self.assertEqual(len(media), 5)

    def test_site_data_matches_authoritative_analysis(self):
        payload = load_window_json(SITE / "paper-data.js", "PAPER_DATA")
        family = pd.read_csv(ANALYSIS / "family_summary.csv")
        target = pd.read_csv(ANALYSIS / "time_to_target_summary.csv")
        speed = pd.read_csv(ANALYSIS / "speed_summary.csv")
        self.assertEqual(payload["summary"]["runs"], 700)
        self.assertEqual(payload["summary"]["maps"], 70)
        self.assertEqual(len(payload["families"]), 7)
        self.assertEqual(len(payload["scenarios"]), 7)
        self.assertEqual(payload["protocol"]["candidate_constructions"], 100000)
        self.assertEqual(
            payload["protocol"]["calibration_configurations_per_algorithm"],
            24,
        )
        expected_s7 = float(
            family.loc[
                family["family"] == "S7", "median_cost_gain_pct"
            ].iloc[0]
        )
        self.assertAlmostEqual(payload["summary"]["s7_gain"], expected_s7, places=2)
        self.assertAlmostEqual(
            payload["summary"]["cpu_speedup_low"],
            float(speed["median_speedup"].min()),
            places=2,
        )
        expected_s7_aco = float(
            target.loc[
                (target["family"] == "S7") & (target["algo"] == "ACO"),
                "attainment_rate",
            ].iloc[0]
        )
        self.assertAlmostEqual(
            payload["summary"]["s7_aco_target_rate"],
            expected_s7_aco,
            places=2,
        )

    def test_site_sections_assets_and_downloads_are_complete(self):
        html = (SITE / "index.html").read_text()
        for section in (
            "start",
            "concepts",
            "problem",
            "mga",
            "aco",
            "protocol",
            "statistics",
            "results",
            "figures",
            "paper",
            "present",
            "viva",
            "evidence",
            "glossary",
        ):
            self.assertIn(f'id="{section}"', html)
        self.assertEqual(len(re.findall(r'data-lesson="[^"]+"', html)), 13)
        for figure in (
            "fig1_problem_protocol.png",
            "fig2_quality_scale.png",
            "fig3_convergence_attainment.png",
            "fig4_runtime_advantage.png",
            "fig5_representative_route.png",
            "supp_dynamic_storyboard.png",
        ):
            self.assertTrue((SITE / "assets" / figure).exists())
        source_downloads = {
            "paper.pdf": MANUSCRIPT / "manuscript.pdf",
            "paper.docx": MANUSCRIPT / "manuscript.docx",
            "supplement.pdf": MANUSCRIPT / "supplement.pdf",
        }
        for name, source in source_downloads.items():
            copied = SITE / "downloads" / name
            self.assertTrue(copied.exists())
            self.assertEqual(sha256(source), sha256(copied))

    def test_beginner_guide_is_complete(self):
        content = load_window_json(
            SITE / "guide-content.js", "GUIDE_CONTENT"
        )
        self.assertGreaterEqual(len(content["glossary"]), 40)
        self.assertGreaterEqual(len(content["quizzes"]), 10)
        self.assertGreaterEqual(len(content["viva"]), 25)
        self.assertEqual(len(content["guideSteps"]), 13)
        self.assertGreaterEqual(len(content["paperSections"]), 9)

        categories = {item["category"] for item in content["viva"]}
        self.assertEqual(
            categories,
            {
                "contribution",
                "method",
                "fairness",
                "statistics",
                "results",
                "scope",
            },
        )
        for question in content["quizzes"]:
            self.assertIn(question["answer"], range(len(question["options"])))
            self.assertTrue(question["explanation"])
        for item in content["viva"]:
            self.assertTrue(item["short"])
            self.assertTrue(item["detail"])
            self.assertTrue(item["avoid"])

    def test_site_interactions_are_wired(self):
        app = (SITE / "app.js").read_text()
        for behavior in (
            "gaaco-guide-mode",
            "gaaco-guide-completed",
            "gaaco-guide-quiz",
            "renderGlossary",
            "renderViva",
            "renderQuiz",
            "updateLessonProgress",
            "populateClaimTable",
        ):
            self.assertIn(behavior, app)


if __name__ == "__main__":
    unittest.main()
