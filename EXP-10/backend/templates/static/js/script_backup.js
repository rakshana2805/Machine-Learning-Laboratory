document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("uploadForm");
    const imageInput = document.getElementById("imageInput");
    const fileName = document.getElementById("fileName");
    const previewWrap = document.getElementById("previewWrap");
    const imagePreview = document.getElementById("imagePreview");
    const removeImage = document.getElementById("removeImage");
    const predictButton = document.getElementById("predictButton");
    const statusBox = document.getElementById("status");
    const emptyResult = document.getElementById("emptyResult");
    const resultCard = document.getElementById("resultCard");
    const predictionElement = document.getElementById("prediction");
    const probabilitiesElement = document.getElementById("probabilities");
    const resultMessage = document.getElementById("resultMessage");

    if (!uploadForm || !imageInput) {
        console.error("Upload form or image input was not found.");
        return;
    }

    const MAX_FILE_SIZE = 10 * 1024 * 1024;
    const ALLOWED_EXTENSIONS = [
        "jpg", "jpeg", "png", "bmp", "tif", "tiff"
    ];

    let selectedFile = null;

    // Create styles for the six-stage interface.
    const stageStyles = document.createElement("style");

    stageStyles.textContent = `
        .pipeline-section {
            margin: 28px 0;
            padding: 24px;
            border: 1px solid #dce8e7;
            border-radius: 18px;
            background: #ffffff;
            color: #173b3b;
        }

        .pipeline-heading {
            margin: 0 0 8px;
            font-size: 22px;
            font-weight: 750;
        }

        .pipeline-description {
            margin: 0 0 22px;
            color: #647776;
            font-size: 14px;
            line-height: 1.6;
        }

        .pipeline-list {
            display: grid;
            gap: 14px;
        }

        .pipeline-step {
            display: grid;
            grid-template-columns: 44px minmax(0, 1fr);
            gap: 14px;
            padding: 16px;
            border: 1px solid #e0e9e8;
            border-radius: 14px;
            background: #fbfdfd;
            transition: border-color .25s, background .25s;
        }

        .pipeline-step.is-processing {
            border-color: #0d9488;
            background: #f0fdfa;
        }

        .pipeline-step.is-completed {
            border-color: #a7d9cc;
            background: #f6fcf9;
        }

        .pipeline-number {
            width: 40px;
            height: 40px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: #e8eeee;
            color: #385b59;
            font-weight: 800;
        }

        .is-processing .pipeline-number {
            background: #0f766e;
            color: white;
        }

        .is-completed .pipeline-number {
            background: #d5f5e7;
            color: #08744c;
        }

        .pipeline-step h4 {
            margin: 2px 0 6px;
            font-size: 16px;
        }

        .pipeline-step p {
            margin: 0;
            color: #647776;
            font-size: 13px;
            line-height: 1.55;
        }

        .pipeline-status {
            display: inline-block;
            margin-top: 8px;
            font-size: 12px;
            font-weight: 750;
            color: #647776;
        }

        .is-processing .pipeline-status {
            color: #0f766e;
        }

        .is-completed .pipeline-status {
            color: #08744c;
        }

        .pipeline-image {
            display: block;
            width: 100%;
            max-width: 360px;
            max-height: 280px;
            margin-top: 14px;
            object-fit: contain;
            background: #eef3f2;
            border-radius: 10px;
            border: 1px solid #e0e9e8;
        }

        .pipeline-progress-track {
            height: 8px;
            overflow: hidden;
            border-radius: 20px;
            background: #e7eeee;
            margin: 16px 0 8px;
        }

        .pipeline-progress-fill {
            height: 100%;
            width: 0%;
            border-radius: inherit;
            background: #0f9488;
            transition: width .3s ease;
        }

        .pipeline-progress-text {
            color: #647776;
            font-size: 13px;
        }

        .pipeline-error {
            color: #b42318;
            background: #fff1f0;
            padding: 12px;
            border-radius: 10px;
        }

        @media (max-width: 520px) {
            .pipeline-section {
                padding: 16px;
            }

            .pipeline-step {
                grid-template-columns: 36px minmax(0, 1fr);
                gap: 10px;
                padding: 12px;
            }

            .pipeline-number {
                width: 34px;
                height: 34px;
            }
        }
    `;

    document.head.appendChild(stageStyles);

    // Create the pipeline interface without changing index.html.
    const pipelineSection = document.createElement("section");
    pipelineSection.className = "pipeline-section";
    pipelineSection.id = "pipelineSection";

    pipelineSection.innerHTML = `
        <h2 class="pipeline-heading">Image Processing Pipeline</h2>
        <p class="pipeline-description">
            Each stage displays its status and actual processing output.
        </p>

        <div class="pipeline-progress-track">
            <div class="pipeline-progress-fill" id="pipelineProgressFill"></div>
        </div>

        <p class="pipeline-progress-text" id="pipelineProgressText">
            Waiting for an image.
        </p>

        <div class="pipeline-list" id="pipelineList"></div>
    `;

    uploadForm.insertAdjacentElement("afterend", pipelineSection);

    const pipelineList = document.getElementById("pipelineList");
    const progressFill = document.getElementById("pipelineProgressFill");
    const progressText = document.getElementById("pipelineProgressText");

    const expectedStages = [
        {
            id: "upload",
            name: "Image Upload",
            description: "Load the original retinal image."
        },
        {
            id: "rgb",
            name: "RGB Conversion",
            description: "Convert the image from BGR to RGB."
        },
        {
            id: "resize",
            name: "Image Resizing",
            description: "Resize the image to 224 × 224 pixels."
        },
        {
            id: "blur",
            name: "Gaussian Blur",
            description: "Apply Gaussian filtering to reduce noise."
        },
        {
            id: "clahe",
            name: "CLAHE Enhancement",
            description: "Enhance local image contrast."
        },
        {
            id: "normalize",
            name: "Normalization",
            description: "Scale pixel values to the range 0–1."
        }
    ];

    function setStatus(message, isError = false) {
        if (!statusBox) return;

        statusBox.textContent = message;
        statusBox.style.color = isError ? "#b42318" : "";
    }

    function resetResults() {
        if (resultCard) resultCard.hidden = true;
        if (emptyResult) emptyResult.hidden = false;

        if (probabilitiesElement) {
            probabilitiesElement.replaceChildren();
        }

        if (predictionElement) {
            predictionElement.textContent = "";
        }

        if (resultMessage) {
            resultMessage.textContent = "";
        }
    }

    function resetPipeline() {
        pipelineList.replaceChildren();
        progressFill.style.width = "0%";
        progressText.textContent = "Waiting for an image.";

        expectedStages.forEach((stage, index) => {
            const card = document.createElement("article");
            card.className = "pipeline-step";
            card.id = `pipeline-step-${stage.id}`;

            const number = document.createElement("div");
            number.className = "pipeline-number";
            number.textContent = String(index + 1).padStart(2, "0");

            const content = document.createElement("div");

            const title = document.createElement("h4");
            title.textContent = stage.name;

            const description = document.createElement("p");
            description.textContent = stage.description;

            const status = document.createElement("span");
            status.className = "pipeline-status";
            status.textContent = "PENDING";
            status.id = `pipeline-status-${stage.id}`;

            content.append(title, description, status);
            card.append(number, content);
            pipelineList.appendChild(card);
        });
    }

    function updateStage(stage, actualStage, status) {
        const card = document.getElementById(
            `pipeline-step-${stage.id}`
        );

        const statusElement = document.getElementById(
            `pipeline-status-${stage.id}`
        );

        if (!card || !statusElement) return;

        card.classList.remove("is-processing", "is-completed");

        if (status === "processing") {
            card.classList.add("is-processing");
            statusElement.textContent = "PROCESSING...";
            progressText.textContent = `Processing: ${stage.name}`;
        }

        if (status === "completed") {
            card.classList.add("is-completed");
            statusElement.textContent = "✓ COMPLETED";

            if (actualStage && actualStage.image) {
                let image = card.querySelector(".pipeline-image");

                if (!image) {
                    image = document.createElement("img");
                    image.className = "pipeline-image";
                    image.alt = `${stage.name} output`;
                    image.loading = "lazy";
                    card.appendChild(image);
                }

                image.src = actualStage.image;
            }
        }

        if (status === "failed") {
            statusElement.textContent = "FAILED";
        }
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function renderProbabilities(probabilities) {
        if (!probabilitiesElement) return;

        probabilitiesElement.replaceChildren();

        Object.entries(probabilities || {}).forEach(([label, value]) => {
            const numericValue = Number(value);

            const row = document.createElement("div");
            row.style.marginBottom = "14px";

            const heading = document.createElement("div");
            heading.style.display = "flex";
            heading.style.justifyContent = "space-between";
            heading.style.gap = "12px";
            heading.style.marginBottom = "6px";

            const name = document.createElement("span");
            name.textContent = label;

            const percentage = document.createElement("strong");
            percentage.textContent = `${numericValue.toFixed(2)}%`;

            heading.append(name, percentage);

            const track = document.createElement("div");
            track.style.height = "8px";
            track.style.background = "#e7eeee";
            track.style.borderRadius = "12px";
            track.style.overflow = "hidden";

            const fill = document.createElement("div");
            fill.style.height = "100%";
            fill.style.width = `${Math.min(100, Math.max(0, numericValue))}%`;
            fill.style.background = "#0f9488";
            fill.style.borderRadius = "12px";

            track.appendChild(fill);
            row.append(heading, track);
            probabilitiesElement.appendChild(row);
        });
    }

    function showPrediction(data) {
        if (emptyResult) emptyResult.hidden = true;
        if (resultCard) resultCard.hidden = false;

        if (predictionElement) {
            predictionElement.textContent = data.prediction ?? "Unavailable";
        }

        renderProbabilities(data.probabilities);

        if (resultMessage) {
            resultMessage.textContent =
                data.message || "Screening prototype result.";
        }
    }

    function getStageForId(stages, id) {
        return stages.find(stage => stage.id === id);
    }

    async function displayStagesSequentially(stages) {
        for (let index = 0; index < expectedStages.length; index++) {
            const expected = expectedStages[index];
            const actual = getStageForId(stages, expected.id);

            updateStage(expected, actual, "processing");

            // Briefly show the active stage before revealing its output.
            await delay(350);

            if (!actual || !actual.image) {
                updateStage(expected, actual, "failed");
                throw new Error(
                    `The processing output for "${expected.name}" is missing.`
                );
            }

            updateStage(expected, actual, "completed");

            const progress = ((index + 1) / expectedStages.length) * 100;
            progressFill.style.width = `${progress}%`;
            progressText.textContent =
                `Completed ${index + 1} of ${expectedStages.length} stages`;

            await delay(250);
        }

        progressText.textContent = "All six processing stages completed.";
    }

    function validateFile(file) {
        if (!file) {
            return "Please select a retinal image first.";
        }

        const extension = file.name.split(".").pop().toLowerCase();

        if (!ALLOWED_EXTENSIONS.includes(extension)) {
            return "Unsupported image format. Choose JPG, JPEG, PNG, BMP or TIFF.";
        }

        if (file.size > MAX_FILE_SIZE) {
            return "Image size must not exceed 10 MB.";
        }

        return null;
    }

    imageInput.addEventListener("change", () => {
        const file = imageInput.files && imageInput.files[0];
        const error = validateFile(file);

        selectedFile = error ? null : file;

        resetPipeline();
        resetResults();

        if (error) {
            if (file) setStatus(error, true);
            return;
        }

        if (!file) {
            if (fileName) fileName.textContent = "";
            if (previewWrap) previewWrap.hidden = true;
            return;
        }

        if (fileName) fileName.textContent = file.name;

        if (previewWrap) previewWrap.hidden = false;

        if (imagePreview) {
            const previewUrl = URL.createObjectURL(file);
            imagePreview.onload = () => URL.revokeObjectURL(previewUrl);
            imagePreview.src = previewUrl;
        }

        setStatus("Image selected. Ready to analyze.");
    });

    if (removeImage) {
        removeImage.addEventListener("click", () => {
            imageInput.value = "";
            selectedFile = null;

            if (fileName) fileName.textContent = "";
            if (previewWrap) previewWrap.hidden = true;
            if (imagePreview) imagePreview.removeAttribute("src");

            resetPipeline();
            resetResults();
            setStatus("Image removed.");
        });
    }

    uploadForm.addEventListener("submit", async event => {
        event.preventDefault();

        const file = selectedFile || (
            imageInput.files && imageInput.files[0]
        );

        const validationError = validateFile(file);

        if (validationError) {
            setStatus(validationError, true);
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        predictButton && (predictButton.disabled = true);

        if (predictButton) {
            predictButton.textContent = "Analyzing...";
        }

        resetPipeline();
        resetResults();
        setStatus("Sending image to the classification model...");

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(
                    data.error || "Image prediction failed."
                );
            }

            if (!Array.isArray(data.stages)) {
                throw new Error(
                    "The server response does not contain processing stages."
                );
            }

            // Display actual stage outputs in sequence.
            await displayStagesSequentially(data.stages);

            showPrediction(data);
            setStatus("Analysis completed successfully.");

        } catch (error) {
            console.error("Prediction error:", error);
            setStatus(error.message || "Prediction failed.", true);
            progressText.textContent =
                "Processing could not be completed.";

        } finally {
            if (predictButton) {
                predictButton.disabled = false;
                predictButton.textContent = "Analyze Image";
            }
        }
    });

    resetPipeline();
    resetResults();
});