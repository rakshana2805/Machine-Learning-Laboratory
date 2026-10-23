
(() => {
  "use strict";

  async function loadEvaluationDashboard() {
    let panel = document.getElementById("liveEvaluationPanel");

    if (!panel) {
      panel = document.createElement("section");
      panel.id = "liveEvaluationPanel";
      panel.style.cssText =
        "margin:24px 0;padding:24px;border:1px solid #ddd;" +
        "border-radius:16px;background:var(--card-bg,#fff);";

      panel.innerHTML = `
        <h2>Model Evaluation</h2>
        <p id="evalMessage">Loading test evaluation results...</p>
        <div id="evalMetrics" style="
          display:grid;
          grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
          gap:12px;margin:20px 0;"></div>
        <div style="
          display:grid;
          grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
          gap:20px;">
          <div>
            <h3>Confusion Matrix Heatmap</h3>
            <img id="evalHeatmap" alt="Confusion matrix heatmap"
              style="width:100%;height:auto;display:none;border-radius:8px;">
          </div>
          <div>
            <h3>Evaluation Metrics</h3>
            <img id="evalMetricsChart" alt="Evaluation metrics bar chart"
              style="width:100%;height:auto;display:none;border-radius:8px;">
          </div>
        </div>
        <h3>Confusion Matrix Values</h3>
        <div id="evalMatrixTable"></div>
        <h3>Class-wise Report</h3>
        <pre id="evalClassReport" style="
          white-space:pre-wrap;overflow-wrap:anywhere;
          padding:12px;background:#f5f5f5;border-radius:8px;"></pre>
      `;

      const existingEvaluation =
        document.getElementById("evaluation") ||
        document.getElementById("evaluationSection") ||
        document.querySelector("[data-section='evaluation']");

      if (existingEvaluation) {
        existingEvaluation.appendChild(panel);
      } else {
        document.querySelector("main")?.appendChild(panel);
        if (!panel.isConnected) document.body.appendChild(panel);
      }
    }

    const message = document.getElementById("evalMessage");

    try {
      const response = await fetch("/api/evaluation", {
        headers: { Accept: "application/json" }
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.message || "Evaluation results unavailable.");
      }

      message.textContent =
        `Evaluation loaded: ${data.image_count ?? data.total_images ?? "—"} test images`;

      const metrics = data.metrics || {};
      const metricContainer = document.getElementById("evalMetrics");

      metricContainer.replaceChildren();

      const metricItems = [
        ["Accuracy", metrics.accuracy],
        ["Macro Precision", metrics.macro_precision],
        ["Macro Recall", metrics.macro_recall],
        ["Macro F1-score", metrics.macro_f1]
      ];

      for (const [name, rawValue] of metricItems) {
        const card = document.createElement("div");
        card.style.cssText =
          "padding:16px;border:1px solid #ddd;border-radius:12px;";

        const value = Number(rawValue);
        const formatted = Number.isFinite(value)
          ? `${value.toFixed(2)}%`
          : "Not available";

        const heading = document.createElement("div");
        heading.textContent = name;
        heading.style.cssText = "font-size:13px;color:#666;";

        const result = document.createElement("strong");
        result.textContent = formatted;
        result.style.cssText = "display:block;font-size:24px;margin-top:8px;";

        card.append(heading, result);
        metricContainer.appendChild(card);
      }

      const heatmap = document.getElementById("evalHeatmap");
      if (data.confusion_matrix_image) {
        heatmap.src = data.confusion_matrix_image;
        heatmap.style.display = "block";
      }

      const metricsChart = document.getElementById("evalMetricsChart");
      if (data.metrics_chart_image) {
        metricsChart.src = data.metrics_chart_image;
        metricsChart.style.display = "block";
      }

      const matrix = data.confusion_matrix;
      const classes = data.classes || [];

      const tableContainer = document.getElementById("evalMatrixTable");
      tableContainer.replaceChildren();

      if (Array.isArray(matrix) && matrix.length) {
        const table = document.createElement("table");
        table.style.cssText =
          "width:100%;border-collapse:collapse;text-align:center;";

        const header = document.createElement("tr");
        const corner = document.createElement("th");
        corner.textContent = "Actual / Predicted";
        header.appendChild(corner);

        classes.forEach((label) => {
          const th = document.createElement("th");
          th.textContent = label;
          header.appendChild(th);
        });
        table.appendChild(header);

        matrix.forEach((row, rowIndex) => {
          const tr = document.createElement("tr");
          const labelCell = document.createElement("th");
          labelCell.textContent = classes[rowIndex] || `Class ${rowIndex + 1}`;
          tr.appendChild(labelCell);

          row.forEach((count) => {
            const td = document.createElement("td");
            td.textContent = String(count);
            td.style.cssText = "padding:12px;border:1px solid #ddd;";
            tr.appendChild(td);
          });
          table.appendChild(tr);
        });

        tableContainer.appendChild(table);
      } else {
        tableContainer.textContent = "Confusion matrix values unavailable.";
      }

      const report = data.classification_report;
      document.getElementById("evalClassReport").textContent =
        typeof report === "string"
          ? report
          : report
            ? JSON.stringify(report, null, 2)
            : "Class-wise report unavailable.";

    } catch (error) {
      message.textContent = `Unable to load evaluation: ${error.message}`;
    }
  }

  document.addEventListener("DOMContentLoaded", loadEvaluationDashboard);
})();
