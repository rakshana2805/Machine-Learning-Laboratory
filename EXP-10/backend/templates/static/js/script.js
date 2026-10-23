"use strict";

document.addEventListener("DOMContentLoaded", () => {
    const $ = (id) => document.getElementById(id);
    const $$ = (selector) => [...document.querySelectorAll(selector)];

    // --------------------------------------------------
    // ELEMENTS
    // --------------------------------------------------

    const uploadForm = $("uploadForm");
    const imageInput = $("imageInput");
    const dropzone = $("dropzone");
    const fileInformation = $("fileInformation");
    const imagePreview = $("imagePreview");
    const fileName = $("fileName");
    const fileSize = $("fileSize");
    const removeImage = $("removeImage");
    const predictButton = $("predictButton");
    const predictButtonText = $("predictButtonText");
    const statusElement = $("status");

    const emptyResult = $("emptyResult");
    const resultCard = $("resultCard");
    const predictionElement = $("prediction");
    const confidenceValue = $("confidenceValue");
    const confidenceBar = $("confidenceBar");
    const probabilitiesElement = $("probabilities");
    const resultMessage = $("resultMessage");
    const predictionSymbol = $("predictionSymbol");
    const rawPrediction = $("rawPrediction");
    const resultFeatureCount = $("resultFeatureCount");

    const stepCounter = $("stepCounter");
    const stepItems = $$(".step-item");

    const historyTableBody = $("historyTableBody");
    const historyCount = $("historyCount");
    const historyMessage = $("historyMessage");
    const refreshHistory = $("refreshHistory");

    const featureTableBody = $("featureTableBody");
    const featureChartCanvas = $("featureChart");
    const featureChartEmpty = $("featureChartEmpty");

    let selectedFile = null;
    let previewObjectUrl = null;
    let isPredicting = false;
    let featureChart = null;
    let accuracyChart = null;
    let lossChart = null;
    let showAllFeatures = false;
    let latestFeatureValues = [];

    const MAX_FILE_SIZE = 10 * 1024 * 1024;

    const ALLOWED_TYPES = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff"
    ];

    const STAGES = [
        { id: "upload", aliases: ["upload", "original", "original_image"] },
        { id: "rgb", aliases: ["rgb", "rgb_conversion", "rgb_conversion_image"] },
        { id: "resize", aliases: ["resize", "resized", "image_resizing"] },
        { id: "blur", aliases: ["blur", "gaussian_blur"] },
        { id: "clahe", aliases: ["clahe", "clahe_enhancement", "enhanced"] },
        { id: "normalize", aliases: ["normalize", "normalized", "normalization"] }
    ];

    // --------------------------------------------------
    // HELPERS
    // --------------------------------------------------

    function makeElement(tag, className = "", text = "") {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== undefined && text !== null) {
            element.textContent = String(text);
        }
        return element;
    }

    function setStatus(message, type = "info") {
        if (!statusElement) return;
        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`;
        statusElement.hidden = false;
    }

    function clearStatus() {
        if (!statusElement) return;
        statusElement.textContent = "";
        statusElement.hidden = true;
        statusElement.className = "status-message";
    }

    function safeNumber(value) {
        if (value === null || value === undefined || value === "") {
            return null;
        }

        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }

    // Supports percentages such as 71.31 and fractions such as 0.7131.
    function formatPercentage(value) {
        const number = safeNumber(value);
        if (number === null) return "—";

        const percent = number >= 0 && number <= 1
            ? number * 100
            : number;

        return `${percent.toFixed(2)}%`;
    }

    function percentageForBar(value) {
        const number = safeNumber(value);
        if (number === null) return 0;

        const percent = number >= 0 && number <= 1
            ? number * 100
            : number;

        return Math.min(100, Math.max(0, percent));
    }

    function setButtonLoading(loading) {
        isPredicting = loading;

        if (predictButton) {
            predictButton.disabled = loading;
            predictButton.setAttribute("aria-busy", String(loading));
        }

        if (predictButtonText) {
            predictButtonText.textContent = loading
                ? "Analyzing..."
                : "Run Analysis";
        }
    }

    function setActiveStep(stepNumber) {
        const step = Math.min(7, Math.max(1, Number(stepNumber) || 1));

        stepItems.forEach((item) => {
            const itemStep = Number(item.dataset.step);
            item.classList.toggle("active", itemStep === step);
            item.classList.toggle("completed", itemStep < step);
        });

        if (stepCounter) {
            stepCounter.textContent = `Step ${step} of 7`;
        }
    }

    function releasePreviewUrl() {
        if (previewObjectUrl) {
            URL.revokeObjectURL(previewObjectUrl);
            previewObjectUrl = null;
        }
    }

    async function fetchJson(url, options = {}) {
        const response = await fetch(url, {
            headers: { Accept: "application/json", ...(options.headers || {}) },
            ...options
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                `The server returned an invalid response for ${url}. Check the Flask terminal.`
            );
        }

        if (!response.ok || data?.success === false) {
            throw new Error(
                data?.error ||
                data?.message ||
                `Request failed: ${url} (HTTP ${response.status})`
            );
        }

        return data;
    }

    // --------------------------------------------------
    // RESET
    // --------------------------------------------------

    function resetResult() {
        if (emptyResult) emptyResult.hidden = false;
        if (resultCard) resultCard.hidden = true;

        if (predictionElement) predictionElement.textContent = "—";
        if (confidenceValue) confidenceValue.textContent = "—";
        if (confidenceBar) confidenceBar.style.width = "0%";
        if (probabilitiesElement) probabilitiesElement.replaceChildren();
        if (rawPrediction) rawPrediction.textContent = "—";
        if (resultFeatureCount) resultFeatureCount.textContent = "—";

        if (resultMessage) {
            resultMessage.textContent = "Prediction details will appear here.";
        }
    }

    function resetStageCards() {
        $$(".stage-card").forEach((card) => {
            card.classList.remove("processing", "completed");

            const imageContainer = card.querySelector(".stage-image");
            const state = card.querySelector(".stage-state");

            if (imageContainer) {
                imageContainer.classList.add("empty-stage");
                imageContainer.replaceChildren(
                    makeElement("span", "", card.dataset.stage || "•")
                );
            }

            if (state) state.textContent = "Waiting for image";
        });
    }

    function resetFeatureData() {
        showAllFeatures = false;
        latestFeatureValues = [];

        if ($("featureCount")) $("featureCount").textContent = "128";
        if ($("featureStatus")) $("featureStatus").textContent = "Pending";

        if (featureTableBody) {
            featureTableBody.replaceChildren();

            const row = document.createElement("tr");
            const cell = makeElement(
                "td",
                "empty-table",
                "Feature values will appear after analysis."
            );
            cell.colSpan = 2;
            row.appendChild(cell);
            featureTableBody.appendChild(row);
        }

        if (featureChart) {
            featureChart.destroy();
            featureChart = null;
        }

        if (featureChartEmpty) {
            featureChartEmpty.hidden = false;
            featureChartEmpty.textContent = "Feature chart will appear after analysis.";
        }

        $("toggleFeaturesButton")?.remove();
    }

    function resetEvaluationData() {
        [
            "evaluationAccuracy",
            "evaluationPrecision",
            "evaluationRecall",
            "evaluationF1"
        ].forEach((id) => {
            if ($(id)) $(id).textContent = "—";
        });

        const body = $("confusionMatrixBody");

        if (body) {
            body.replaceChildren();

            ["Glaucoma", "Normal"].forEach((label) => {
                const row = document.createElement("tr");
                row.appendChild(makeElement("th", "", label));
                row.appendChild(makeElement("td", "", "—"));
                row.appendChild(makeElement("td", "", "—"));
                body.appendChild(row);
            });
        }

        if ($("evaluationNote")) {
            $("evaluationNote").textContent = "Loading saved evaluation results...";
        }
    }

    function resetTrainingData() {
        [
            "trainingAccuracy",
            "validationAccuracy",
            "trainingEpochs"
        ].forEach((id) => {
            if ($(id)) $(id).textContent = "—";
        });

        if (accuracyChart) {
            accuracyChart.destroy();
            accuracyChart = null;
        }

        if (lossChart) {
            lossChart.destroy();
            lossChart = null;
        }

        if ($("accuracyChartEmpty")) {
            $("accuracyChartEmpty").hidden = false;
        }

        if ($("lossChartEmpty")) {
            $("lossChartEmpty").hidden = false;
        }
    }

    // --------------------------------------------------
    // FILE UPLOAD
    // --------------------------------------------------

    function selectFile(file) {
        if (!file || isPredicting) return;

        clearStatus();

        if (!ALLOWED_TYPES.includes(file.type)) {
            setStatus(
                "Unsupported format. Select a JPG, PNG, WEBP, BMP or TIFF image.",
                "error"
            );
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            setStatus("Image size must not exceed 10 MB.", "error");
            return;
        }

        releasePreviewUrl();
        selectedFile = file;
        previewObjectUrl = URL.createObjectURL(file);

        if (imagePreview) imagePreview.src = previewObjectUrl;
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = formatFileSize(file.size);

        if (fileInformation) fileInformation.hidden = false;
        if (dropzone) dropzone.hidden = true;

        resetResult();
        resetStageCards();
        resetFeatureData();
        resetEvaluationData();
        resetTrainingData();

        setActiveStep(1);
        setStatus("Image selected. Ready to run analysis.", "success");
    }

    function clearSelectedFile() {
        if (isPredicting) return;

        selectedFile = null;
        releasePreviewUrl();

        if (imageInput) imageInput.value = "";
        if (imagePreview) imagePreview.removeAttribute("src");
        if (fileInformation) fileInformation.hidden = true;
        if (dropzone) dropzone.hidden = false;

        clearStatus();
        resetResult();
        resetStageCards();
        resetFeatureData();
        resetEvaluationData();
        resetTrainingData();
        setActiveStep(1);
    }

    imageInput?.addEventListener("change", (event) => {
        const file = event.target.files?.[0];
        if (file) selectFile(file);
    });

    removeImage?.addEventListener("click", clearSelectedFile);

    dropzone?.addEventListener("click", (event) => {
        if (event.target.closest("button, input, a")) return;
        imageInput?.click();
    });

    dropzone?.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            imageInput?.click();
        }
    });

    dropzone?.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone?.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });

    dropzone?.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("drag-over");

        const file = event.dataTransfer?.files?.[0];
        if (file) selectFile(file);
    });

    // --------------------------------------------------
    // PREPROCESSING STAGES
    // --------------------------------------------------

    function getStageImage(stages, stageId) {
        if (!stages) return null;

        const definition = STAGES.find((item) => item.id === stageId);
        const aliases = definition?.aliases || [stageId];

        if (Array.isArray(stages)) {
            const stage = stages.find((item) => {
                const id = String(item?.id || item?.name || item?.stage || "")
                    .toLowerCase()
                    .replaceAll("-", "_")
                    .replaceAll(" ", "_");

                return aliases.includes(id);
            });

            if (stage) {
                return stage.image ||
                    stage.data_url ||
                    stage.dataUrl ||
                    stage.url ||
                    stage.base64 ||
                    null;
            }

            return null;
        }

        for (const key of aliases) {
            const value = stages[key];

            if (typeof value === "string") return value;

            if (value && typeof value === "object") {
                return value.image ||
                    value.data_url ||
                    value.dataUrl ||
                    value.url ||
                    value.base64 ||
                    null;
            }
        }

        return null;
    }

    function normalizeImageSource(source) {
        if (typeof source !== "string" || !source.trim()) return null;

        const value = source.trim();

        if (
            value.startsWith("data:image/") ||
            value.startsWith("blob:") ||
            value.startsWith("https://") ||
            value.startsWith("http://") ||
            value.startsWith("/")
        ) {
            return value;
        }

        if (/^[A-Za-z0-9+/=\r\n]+$/.test(value)) {
            return `data:image/jpeg;base64,${value}`;
        }

        return null;
    }

    function renderSingleStage(card, source) {
        const imageContainer = card.querySelector(".stage-image");
        const state = card.querySelector(".stage-state");

        if (!imageContainer || !state) return;

        const imageSource = normalizeImageSource(source);

        if (!imageSource) {
            state.textContent = "No stage image returned";
            card.classList.remove("processing");
            return;
        }

        const image = document.createElement("img");
        image.alt = `${card.querySelector("h4")?.textContent || "Image"} stage`;
        image.loading = "lazy";
        image.src = imageSource;

        image.addEventListener("load", () => {
            card.classList.remove("processing");
            card.classList.add("completed");
            state.textContent = "Completed";
        });

        image.addEventListener("error", () => {
            card.classList.remove("processing");
            state.textContent = "Image could not be displayed";
        });

        imageContainer.classList.remove("empty-stage");
        imageContainer.replaceChildren(image);
    }

    async function displayPreprocessingStages(stages) {
        let successfulImages = 0;

        for (const stage of STAGES) {
            const card = document.querySelector(
                `.stage-card[data-stage="${stage.id}"]`
            );

            if (!card) continue;

            card.classList.add("processing");

            const state = card.querySelector(".stage-state");
            if (state) state.textContent = "Loading stage...";

            const source = getStageImage(stages, stage.id);

            if (source) {
                successfulImages++;
                renderSingleStage(card, source);
            } else {
                card.classList.remove("processing");
                if (state) state.textContent = "No stage image returned";
            }
        }

        if (successfulImages > 0) setActiveStep(2);

        if (successfulImages !== STAGES.length) {
            console.warn(
                `Preprocessing images received: ${successfulImages}/${STAGES.length}.`,
                stages
            );
        }
    }

    // --------------------------------------------------
    // PREDICTION
    // --------------------------------------------------

    function normalizeProbabilities(probabilities) {
        if (!probabilities) return [];

        if (Array.isArray(probabilities)) {
            return probabilities.map((item, index) => {
                if (item && typeof item === "object") {
                    return {
                        label: String(
                            item.label ?? item.class ?? item.name ?? `Class ${index + 1}`
                        ),
                        value: safeNumber(
                            item.probability ?? item.confidence ?? item.value
                        )
                    };
                }

                return {
                    label: `Class ${index + 1}`,
                    value: safeNumber(item)
                };
            }).filter((item) => item.value !== null);
        }

        if (typeof probabilities === "object") {
            return Object.entries(probabilities).map(([label, value]) => ({
                label,
                value: safeNumber(
                    typeof value === "object"
                        ? value.probability ?? value.confidence ?? value.value
                        : value
                )
            })).filter((item) => item.value !== null);
        }

        return [];
    }

    function renderProbabilities(probabilities) {
        if (!probabilitiesElement) return;

        probabilitiesElement.replaceChildren();
        const entries = normalizeProbabilities(probabilities);

        if (!entries.length) {
            probabilitiesElement.appendChild(
                makeElement("p", "muted-text", "Class probabilities are unavailable.")
            );
            return;
        }

        entries.forEach((entry) => {
            const row = makeElement("div", "probability-row");
            const name = makeElement("span", "probability-name", entry.label);
            const track = makeElement("div", "probability-track");
            const fill = makeElement("div", "probability-fill");
            const value = makeElement(
                "span",
                "probability-value",
                formatPercentage(entry.value)
            );

            fill.style.width = `${percentageForBar(entry.value)}%`;
            track.appendChild(fill);
            row.append(name, track, value);
            probabilitiesElement.appendChild(row);
        });
    }

    function renderPrediction(data) {
        if (emptyResult) emptyResult.hidden = true;
        if (resultCard) resultCard.hidden = false;

        const predictedClass = String(
            data.prediction ?? data.predicted_class ?? "Unknown"
        );

        if (predictionElement) predictionElement.textContent = predictedClass;

        if (rawPrediction) {
            rawPrediction.textContent = String(
                data.raw_prediction ?? predictedClass
            );
        }

        const confidence = safeNumber(data.confidence);

        if (confidenceValue) {
            confidenceValue.textContent = confidence === null
                ? "—"
                : formatPercentage(confidence);
        }

        if (confidenceBar) {
            confidenceBar.style.width = `${percentageForBar(confidence)}%`;
        }

        if (resultMessage) {
            resultMessage.textContent = data.message ||
                "Prediction returned by the saved classification model.";
        }

        if (resultFeatureCount) {
            resultFeatureCount.textContent = data.feature_count ?? "—";
        }

        const isGlaucoma = predictedClass.toLowerCase().includes("glaucoma");

        if (predictionSymbol) {
            predictionSymbol.textContent = isGlaucoma ? "!" : "✓";
            predictionSymbol.style.color = isGlaucoma ? "#b7791f" : "#16865c";
            predictionSymbol.style.background = isGlaucoma ? "#fff4dc" : "#e9f8f0";
        }

        renderProbabilities(data.probabilities);
        setActiveStep(5);
    }

    // --------------------------------------------------
    // FEATURE TABLE + CHART
    // --------------------------------------------------

    function ensureFeatureToggleButton() {
        let button = $("toggleFeaturesButton");
        if (button) return button;

        const table = featureTableBody?.closest("table");
        if (!table) return null;

        button = makeElement("button", "secondary-button", "Show all 128");
        button.id = "toggleFeaturesButton";
        button.type = "button";
        button.style.marginTop = "12px";
        table.insertAdjacentElement("afterend", button);

        button.addEventListener("click", () => {
            showAllFeatures = !showAllFeatures;
            renderFeatureTable(latestFeatureValues);
        });

        return button;
    }

    function renderFeatureTable(values) {
        if (!featureTableBody) return;

        featureTableBody.replaceChildren();

        const limit = showAllFeatures
            ? values.length
            : Math.min(10, values.length);

        const fragment = document.createDocumentFragment();

        for (let index = 0; index < limit; index++) {
            const row = document.createElement("tr");
            row.appendChild(makeElement("td", "", index + 1));

            const value = values[index];

            row.appendChild(
                makeElement("td", "", value === null ? "—" : value.toFixed(6))
            );

            fragment.appendChild(row);
        }

        featureTableBody.appendChild(fragment);

        const button = ensureFeatureToggleButton();

        if (button) {
            button.hidden = values.length <= 10;
            button.textContent = showAllFeatures
                ? "Show first 10"
                : `Show all ${values.length}`;
        }
    }

    function renderFeatureData(data) {
        const rawValues = data?.features ?? data?.feature_values;

        if (!Array.isArray(rawValues) || rawValues.length === 0) {
            if ($("featureStatus")) $("featureStatus").textContent = "No feature data returned";
            if ($("featureCount")) {
                $("featureCount").textContent = data?.feature_count ?? "128";
            }
            if (featureChartEmpty) {
                featureChartEmpty.hidden = false;
                featureChartEmpty.textContent = "Feature data was not returned by the backend.";
            }
            console.warn("No feature array in prediction response:", data);
            return;
        }

        latestFeatureValues = rawValues.map((value) => {
            if (value && typeof value === "object") {
                return safeNumber(value.value);
            }
            return safeNumber(value);
        });

        showAllFeatures = false;

        if ($("featureCount")) {
            $("featureCount").textContent =
                data.feature_count ?? latestFeatureValues.length;
        }

        if ($("featureStatus")) {
            $("featureStatus").textContent =
                `${latestFeatureValues.length} loaded`;
        }

        renderFeatureTable(latestFeatureValues);

        if (!featureChartCanvas || typeof Chart === "undefined") {
            if (featureChartEmpty) {
                featureChartEmpty.hidden = false;
                featureChartEmpty.textContent =
                    "Feature values loaded, but Chart.js is not available.";
            }
            return;
        }

        if (featureChart) featureChart.destroy();

        featureChart = new Chart(featureChartCanvas, {
            type: "line",
            data: {
                labels: latestFeatureValues.map((_, index) => index + 1),
                datasets: [{
                    label: "Feature value",
                    data: latestFeatureValues.map((value) => value ?? 0),
                    borderColor: "#4f65e9",
                    backgroundColor: "rgba(79, 101, 233, 0.12)",
                    fill: true,
                    tension: 0.2,
                    pointRadius: 1.5,
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: "Feature index" } },
                    y: { title: { display: true, text: "Feature value" } }
                }
            }
        });

        if (featureChartEmpty) featureChartEmpty.hidden = true;
        setActiveStep(3);
    }

    // --------------------------------------------------
    // TRAINING HISTORY
    // Only displays actual history supplied by the backend.
    // --------------------------------------------------

    function createHistoryChart(canvasId, emptyId, history, series) {
        const canvas = $(canvasId);
        const empty = $(emptyId);

        if (
            !canvas ||
            typeof Chart === "undefined" ||
            !Array.isArray(history.labels) ||
            history.labels.length === 0
        ) {
            if (empty) {
                empty.hidden = false;
                empty.textContent = "Training history is not available.";
            }
            return null;
        }

        const datasets = series
            .filter((item) => Array.isArray(history[item.key]))
            .map((item) => ({
                label: item.label,
                data: history[item.key],
                borderColor: item.color,
                backgroundColor: item.color,
                tension: 0.25,
                fill: false
            }));

        if (!datasets.length) {
            if (empty) {
                empty.hidden = false;
                empty.textContent = "No training metrics were returned.";
            }
            return null;
        }

        if (empty) empty.hidden = true;

        return new Chart(canvas, {
            type: "line",
            data: { labels: history.labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, position: "bottom" } },
                scales: { y: { beginAtZero: true } }
            }
        });
    }

    function renderTrainingHistory(history) {
        if (!history || typeof history !== "object") {
            if ($("accuracyChartEmpty")) {
                $("accuracyChartEmpty").textContent =
                    "Training history was not returned by the backend.";
                $("accuracyChartEmpty").hidden = false;
            }
            if ($("lossChartEmpty")) {
                $("lossChartEmpty").textContent =
                    "Training history was not returned by the backend.";
                $("lossChartEmpty").hidden = false;
            }
            return;
        }

        if (accuracyChart) accuracyChart.destroy();
        if (lossChart) lossChart.destroy();

        const labels = history.epochs ?? history.labels ?? [];

        const chartHistory = {
            labels,
            train_accuracy: history.train_accuracy ?? history.accuracy,
            val_accuracy: history.val_accuracy ?? history.validation_accuracy,
            train_loss: history.train_loss ?? history.loss,
            val_loss: history.val_loss ?? history.validation_loss
        };

        accuracyChart = createHistoryChart(
            "accuracyChart",
            "accuracyChartEmpty",
            chartHistory,
            [
                { key: "train_accuracy", label: "Training accuracy", color: "#4f65e9" },
                { key: "val_accuracy", label: "Validation accuracy", color: "#19a878" }
            ]
        );

        lossChart = createHistoryChart(
            "lossChart",
            "lossChartEmpty",
            chartHistory,
            [
                { key: "train_loss", label: "Training loss", color: "#4f65e9" },
                { key: "val_loss", label: "Validation loss", color: "#e58c39" }
            ]
        );

        const trainAccuracy = chartHistory.train_accuracy;
        const valAccuracy = chartHistory.val_accuracy;

        if (Array.isArray(trainAccuracy) && trainAccuracy.length && $("trainingAccuracy")) {
            $("trainingAccuracy").textContent =
                formatPercentage(trainAccuracy[trainAccuracy.length - 1]);
        }

        if (Array.isArray(valAccuracy) && valAccuracy.length && $("validationAccuracy")) {
            $("validationAccuracy").textContent =
                formatPercentage(valAccuracy[valAccuracy.length - 1]);
        }

        if ($("trainingEpochs")) {
            $("trainingEpochs").textContent = labels.length || "—";
        }
    }

    // --------------------------------------------------
    // EVALUATION RESULTS
    // Loads actual metrics from /api/evaluation separately.
    // --------------------------------------------------

    function unwrapEvaluation(payload) {
        if (!payload || typeof payload !== "object") return null;

        // Support both a direct result and a nested result object.
        return payload.evaluation ??
            payload.results ??
            payload;
    }

    function renderEvaluation(evaluation) {
        if (!evaluation || typeof evaluation !== "object") return;

        const metrics = evaluation.metrics ?? evaluation;
        const report = evaluation.classification_report ?? evaluation.report ?? {};

        const precision = metrics.macro_precision ??
            metrics.precision ??
            report["macro avg"]?.precision;

        const recall = metrics.macro_recall ??
            metrics.recall ??
            report["macro avg"]?.recall;

        const f1 = metrics.macro_f1 ??
            metrics.f1_score ??
            metrics.f1 ??
            report["macro avg"]?.["f1-score"] ??
            report["macro avg"]?.f1_score;

        const metricMap = {
            evaluationAccuracy: metrics.accuracy ?? metrics.test_accuracy,
            evaluationPrecision: precision,
            evaluationRecall: recall,
            evaluationF1: f1
        };

        Object.entries(metricMap).forEach(([id, value]) => {
            if ($(id) && value !== undefined && value !== null) {
                $(id).textContent = formatPercentage(value);
            }
        });

        const matrix = evaluation.confusion_matrix ?? metrics.confusion_matrix;
        const labels = evaluation.classes ?? evaluation.class_names ??
            metrics.classes ?? ["Glaucoma", "Normal"];

        const body = $("confusionMatrixBody");

        if (body && Array.isArray(matrix) && matrix.length >= 2) {
            body.replaceChildren();

            matrix.slice(0, 2).forEach((rowValues, rowIndex) => {
                const row = document.createElement("tr");

                row.appendChild(
                    makeElement("th", "", labels[rowIndex] ?? `Class ${rowIndex + 1}`)
                );

                (Array.isArray(rowValues) ? rowValues : []).slice(0, 2).forEach(
                    (value, colIndex) => {
                        const cell = makeElement("td", "", value);
                        cell.classList.add(
                            rowIndex === colIndex ? "diagonal" : "off-diagonal"
                        );
                        row.appendChild(cell);
                    }
                );

                body.appendChild(row);
            });
        }

        const dataset = evaluation.dataset ??
            evaluation.test_dataset ??
            evaluation.test_path;

        if ($("evaluationNote")) {
            $("evaluationNote").textContent = dataset
                ? `Evaluation dataset: ${dataset}`
                : "Showing saved test-set evaluation results.";
        }

        // Show actual evaluation images if the API provides them.
        const heatmap = evaluation.confusion_matrix_image ??
            evaluation.heatmap_image;
        const metricsChart = evaluation.metrics_chart_image ??
            evaluation.metrics_image;

        showEvaluationImage("confusionMatrixImage", heatmap, "Confusion matrix heatmap");
        showEvaluationImage("evaluationMetricsImage", metricsChart, "Evaluation metrics chart");

        setActiveStep(6);
    }

    function showEvaluationImage(id, source, alt) {
        const existing = $(id);
        if (!existing || !source) return;

        existing.src = source;
        existing.alt = alt;
        existing.hidden = false;
    }

    async function loadEvaluation() {
        try {
            const data = await fetchJson("/api/evaluation");
            renderEvaluation(unwrapEvaluation(data));
        } catch (error) {
            console.warn("Evaluation results could not be loaded:", error);

            if ($("evaluationNote")) {
                $("evaluationNote").textContent =
                    "Could not load evaluation results. Check that the evaluation script has been run and /api/evaluation is available.";
            }
        }
    }

    // --------------------------------------------------
    // PREDICTION HISTORY
    // --------------------------------------------------

    function renderHistory(data) {
        const records = Array.isArray(data)
            ? data
            : data?.history ?? data?.records;

        if (!Array.isArray(records)) {
            if (historyMessage) {
                historyMessage.textContent =
                    "History API did not return a records list.";
            }
            return;
        }

        if (historyTableBody) historyTableBody.replaceChildren();
        if (historyCount) {
            historyCount.textContent = String(data?.count ?? records.length);
        }

        if (records.length === 0) {
            if (historyTableBody) {
                const row = document.createElement("tr");
                const cell = makeElement("td", "empty-table", "No prediction history yet.");
                cell.colSpan = 4;
                row.appendChild(cell);
                historyTableBody.appendChild(row);
            }

            if (historyMessage) {
                historyMessage.textContent = data?.message || "No saved predictions found.";
            }
            return;
        }

        records.forEach((record) => {
            const row = document.createElement("tr");

            row.appendChild(makeElement(
                "td",
                "",
                record.timestamp ?? record.created_at ?? record.date_time ?? "—"
            ));

            row.appendChild(makeElement(
                "td",
                "",
                record.filename ?? record.file_name ?? "—"
            ));

            const classCell = document.createElement("td");
            classCell.appendChild(
                makeElement(
                    "span",
                    "prediction-pill",
                    record.prediction ?? record.predicted_class ?? "—"
                )
            );
            row.appendChild(classCell);

            row.appendChild(makeElement(
                "td",
                "",
                record.confidence == null ? "—" : formatPercentage(record.confidence)
            ));

            historyTableBody?.appendChild(row);
        });

        if (historyMessage) {
            historyMessage.textContent =
                "Displaying saved prediction records from the database.";
        }
    }

    async function loadHistory() {
        if (historyMessage) historyMessage.textContent = "Loading prediction history...";

        try {
            const data = await fetchJson("/api/history");
            renderHistory(data);
        } catch (error) {
            console.warn("History unavailable:", error);

            if (historyMessage) {
                historyMessage.textContent =
                    "Could not load prediction history. Check the /api/history endpoint and SQLite database.";
            }
        }
    }

    refreshHistory?.addEventListener("click", loadHistory);

    // --------------------------------------------------
    // RUN ANALYSIS
    // --------------------------------------------------

    uploadForm?.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (isPredicting) return;

        if (!selectedFile) {
            setStatus("Please select an image before running analysis.", "error");
            return;
        }

        clearStatus();
        setButtonLoading(true);
        setActiveStep(2);

        resetResult();
        resetStageCards();
        resetFeatureData();

        setStatus(
            "Uploading image and running preprocessing, feature extraction and prediction...",
            "info"
        );

        try {
            const formData = new FormData();
            formData.append("image", selectedFile, selectedFile.name);

            const data = await fetchJson("/api/predict", {
                method: "POST",
                body: formData
            });

            // 1. Classification result
            renderPrediction(data);

            // 2. Preprocessing visualization
            await displayPreprocessingStages(data.stages);

            // 3. Extracted features
            renderFeatureData(data);

            // 4. Training curves, only if the backend returns real history
            renderTrainingHistory(data.training_history ?? data.trainingHistory);

            // 5. Load persisted test evaluation and history independently.
            await Promise.allSettled([
                loadEvaluation(),
                loadHistory()
            ]);

            setStatus(
                "Analysis completed. Prediction and available backend results have been loaded.",
                "success"
            );
        } catch (error) {
            console.error("Analysis error:", error);

            setStatus(
                error.message || "Analysis failed. Check the Flask terminal.",
                "error"
            );

            setActiveStep(1);
        } finally {
            setButtonLoading(false);
        }
    });

    // --------------------------------------------------
    // INITIALIZE
    // --------------------------------------------------

    resetStageCards();
    resetResult();
    resetFeatureData();
    resetTrainingData();
    resetEvaluationData();
    setActiveStep(1);

    // Load previously saved information when the page opens.
    loadHistory();
    loadEvaluation();
});