import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
NAVY = "102B46"


def paragraph(doc, text, style=None):
    return doc.add_paragraph(text, style)


def heading(doc, number, title):
    doc.add_heading(title, 0)


def picture(doc, name, caption, width=6.8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(ROOT / "submission/evidence" / name), width=Inches(width))
    paragraph(doc, caption, "Caption")


def main():
    validation = json.loads((ROOT / "reports/validation.json").read_text())
    evaluation = json.loads((ROOT / "reports/evaluation.json").read_text())
    registry = json.loads((ROOT / "reports/registry.json").read_text())
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.65)
    section.left_margin = section.right_margin = Inches(0.7)
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in [("Title", 20), ("Heading 1", 14), ("Heading 2", 12)]:
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(NAVY)
        style.font.bold = True
    doc.styles["Subtitle"].font.color.rgb = RGBColor(0, 0, 0)
    doc.styles["Caption"].font.size = Pt(8)
    doc.styles["Caption"].font.color.rgb = RGBColor(0, 0, 0)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    heading(doc, 1, "Flood Early Warning\nfor Sindh and Punjab")
    paragraph(doc, "A reproducible machine learning pipeline for Indus river reaches", "Subtitle")
    doc.add_heading("Group members", 1)
    members = doc.add_table(rows=5, cols=2)
    members.style = "Table Grid"
    members.columns[0].width = Inches(4.8)
    members.columns[1].width = Inches(2.0)
    members.cell(0, 0).text = "Name"
    members.cell(0, 1).text = "Roll No."
    for cell in members.rows[0].cells:
        cell.paragraphs[0].runs[0].font.bold = True
    for row in members.rows:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.space_after = Pt(3)
            cell.paragraphs[0].paragraph_format.space_before = Pt(3)
    stats = doc.add_table(rows=1, cols=3)
    stats.style = "Table Grid"
    stats.alignment = WD_TABLE_ALIGNMENT.CENTER
    stats.autofit = False
    for column in stats.columns:
        column.width = Inches(2.29)
    for cell, text in zip(
        stats.rows[0].cells,
        [
            f"{validation['rows']:,}\nreach-day records",
            "4\nriver reaches",
            "3–7 days\nprediction lead times",
        ],
    ):
        cell.text = text
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(0, 0, 0)
            run.font.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading("Project links", 1)
    for label, url in [
        ("GitHub", "https://github.com/Muzammil-Aleem/sindh-punjab-flood-warning"),
        ("DagsHub", "https://dagshub.com/Muzammil-Aleem/sindh-punjab-flood-warning"),
        ("MLflow tracking", "https://dagshub.com/Muzammil-Aleem/sindh-punjab-flood-warning.mlflow"),
    ]:
        p = paragraph(doc, label + "  ")
        p.add_run(url).font.size = Pt(9)
    doc.add_heading("Data ingestion and validation", 1)
    paragraph(
        doc,
        f"The dataset spans {validation['start']} to {validation['end']}. GloFAS historical discharge is retrieved as GRIB through Copernicus and converted to daily point series. CHIRPS rainfall uses ClimateSERV. Hourly ERA5 temperature and topsoil moisture use Open-Meteo’s archive with the ERA5 model explicitly selected and elevation downscaling disabled. All observations align to UTC calendar days.",
    )
    paragraph(
        doc,
        "Official ECMWF two-year discharge thresholds are pinned by source version and SHA256. Validation rejects schema mismatches, null or infinite values, duplicate reach-days, missing dates and invalid physical ranges. The final dataset contains no missing values or duplicate reach-days.",
    )
    doc.add_heading("Scope and interpretation", 1)
    paragraph(
        doc,
        "Labels represent GloFAS reanalysis threshold exceedances, rather than independently observed inundation. This retrospective study does not establish operational warning skill. Archived GloFAS forecast comparisons and deployment remain outside this milestone.",
    )
    doc.add_page_break()
    heading(doc, 2, "Features and experiments")
    doc.add_heading("Chronological model development", 1)
    paragraph(
        doc,
        "Training uses 2010–2014, validation uses 2015–2021, and the held-out test uses 2022. Targets crossing split boundaries are purged. Features include discharge, rainfall, temperature and soil moisture lags; rolling rainfall and discharge sums; discharge changes; seasonality; and the prediction horizon. Scalers fit only on training observations.",
    )
    paragraph(
        doc,
        "The pipeline also exports 30-day, four-variable sequence tensors with five aligned horizon labels for later temporal models. Persistence, class-weighted logistic regression and a class-weighted random forest form the current comparison. Decision thresholds and the trained-model winner are selected using validation F1.",
    )
    picture(
        doc,
        "runs.png",
        "Figure 1. Three completed experiment runs with validation and held-out metrics in MLflow.",
    )
    doc.add_heading("Remote MLflow tracking", 1)
    paragraph(
        doc,
        "Each run records model parameters, split settings, dataset fingerprints, validation and test metrics, a model signature, dependency versions, source manifests and evaluation plots. Precision, recall and PR-AUC accompany F1, with breakdowns by lead time and river reach.",
    )
    paragraph(
        doc,
        "Screenshots use the official MLflow frontend connected read-only to hosted DagsHub APIs. There is no local tracking datastore.",
    )
    doc.add_page_break()
    heading(doc, 3, "Evaluation and registration")
    paragraph(
        doc,
        "2022 held-out performance, pooled across prediction horizons from three through seven days.",
    )
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for column, width in zip(table.columns, [2.35, 1.12, 1.12, 1.12, 1.12]):
        column.width = Inches(width)
    for cell, value in zip(table.rows[0].cells, ["Model", "F1", "Precision", "Recall", "PR-AUC"]):
        cell.text = value
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(0, 0, 0)
            run.font.bold = True
    for i, (name, result) in enumerate(evaluation["models"].items()):
        values = [
            name.replace("_", " "),
            *[f"{result['test'][key]:.4f}" for key in ["f1", "precision", "recall", "pr_auc"]],
        ]
        for cell, value in zip(table.add_row().cells, values):
            cell.text = value
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
    doc.add_heading("Validation-selected model", 1)
    paragraph(
        doc,
        f"Random forest is registered as version {registry['version']} of {registry['name']} in remote Staging. Its serialized classifier preserves the selected decision threshold of {registry['decision_threshold']:.2f}. Downloading the registered model reproduced the local model’s predictions on all 7,200 held-out samples.",
    )
    paragraph(
        doc,
        "Persistence outperforms random forest on 2022 test F1 (0.444 versus 0.400). Registration follows validation selection and does not imply test-set superiority. Only Kotri exceeds the fixed thresholds in 2022; per-reach reports include positive counts and mark PR-AUC undefined for single-class groups.",
    )
    picture(
        doc,
        "registry.png",
        "Figure 2. The actual registered model, version 1, in Staging on the hosted registry.",
    )
    doc.add_page_break()
    heading(doc, 4, "Reproducibility and DVC")
    paragraph(
        doc,
        "Seven named stages retrieve thresholds, ingest observations, validate data, engineer features, train models, evaluate the hold-out year and register the validation winner. DVC versions point observations, validated series, features, models and evaluation artifacts using the configured DagsHub remote. Original GRIB files remain in a reusable download cache and are fingerprinted in the manifest.",
    )
    picture(doc, "dvc_dag.png", "Figure 3. Actual dependency graph produced by dvc dag.", width=2.9)
    doc.add_heading("Run and verification", 1)
    paragraph(
        doc,
        "make run executes dvc repro; make push uploads tracked outputs. A second pipeline run skipped all unchanged stages, and DVC confirmed that the local cache and remote storage were in sync. The environment is pinned and training uses fixed random seeds.",
    )
    paragraph(
        doc,
        "All 54 tests pass, covering validation, causal features, future-label alignment, chronological splits, sequence shapes, prediction behavior and serialization across processes. Isolated synthetic fixtures support tests; reported experiment results use real source data. Credentials, raw data and model weights are excluded from Git.",
    )
    doc.core_properties.title = "Flood Early Warning for Sindh and Punjab"
    doc.core_properties.subject = "Core ML pipeline milestone"
    target = ROOT / "submission/flood-warning.docx"
    doc.save(target)
    print(target)


if __name__ == "__main__":
    main()
