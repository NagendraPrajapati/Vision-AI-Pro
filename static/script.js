
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const fileName = document.getElementById("fileName");
const uploadForm = document.getElementById("uploadForm");
const analyzeButton = document.getElementById("analyzeButton");
const themeToggle = document.getElementById("themeToggle");

// Image Preview
if (imageInput) {
    imageInput.addEventListener("change", function () {
        const file = this.files[0];

        if (!file) return;

        const allowedTypes = [
            "image/jpeg",
            "image/png",
            "image/webp"
        ];

        if (!allowedTypes.includes(file.type)) {
            alert("Please select JPG, PNG or WEBP image.");
            this.value = "";
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert("Image size must be below 10 MB.");
            this.value = "";
            return;
        }

        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        fileName.textContent = file.name;
    });
}

// Loading Animation
if (uploadForm) {
    uploadForm.addEventListener("submit", function () {
        analyzeButton.textContent = "⏳ AI is analyzing...";
        analyzeButton.disabled = true;
    });
}

// Dark / Light Mode
if (themeToggle) {
    themeToggle.addEventListener("click", function () {
        document.body.classList.toggle("light-mode");

        if (document.body.classList.contains("light-mode")) {
            themeToggle.textContent = "☾ Dark Mode";
        } else {
            themeToggle.textContent = "☀ Light Mode";
        }
    });
}

// Confidence Slider
const confidenceSlider =
    document.getElementById("confidenceSlider");

const confidenceValue =
    document.getElementById("confidenceValue");

if (confidenceSlider && confidenceValue) {
    confidenceSlider.addEventListener("input", function () {
        confidenceValue.textContent = this.value + "%";
    });
}

// Object Search
const objectSearch = document.getElementById("objectSearch");
const objectRows = document.querySelectorAll(".object-row");

if (objectSearch) {
    objectSearch.addEventListener("input", function () {
        const search = this.value.toLowerCase();

        objectRows.forEach(row => {
            const name = row.dataset.object || "";

            row.style.display = name.includes(search)
                ? "flex"
                : "none";
        });
    });
}

// Hindi Voice Assistant

const voiceButton = document.getElementById("voiceButton");
const stopVoiceButton = document.getElementById("stopVoiceButton");

const hindiNames = {
    person: "व्यक्ति",
    car: "कार",
    bus: "बस",
    bicycle: "साइकिल",
    motorcycle: "मोटरसाइकिल",
    dog: "कुत्ता",
    cat: "बिल्ली",
    bird: "पक्षी",
    bottle: "बोतल",
    chair: "कुर्सी",
    laptop: "लैपटॉप",
    cell_phone: "मोबाइल फोन"
};

if (voiceButton) {
    voiceButton.addEventListener("click", function () {

        const objectRows = document.querySelectorAll(".object-row");

        if (objectRows.length === 0) {
            alert("पहले किसी फोटो को Analyze करें।");
            return;
        }

        const counts = {};

        objectRows.forEach(row => {
            const name = row.dataset.object;
            if (name) {
                counts[name] = (counts[name] || 0) + 1;
            }
        });

        let message = "नमस्कार! आपका ऑब्जेक्ट एनालिसिस पूरा हो गया है। ";

        Object.entries(counts).forEach(([name, count]) => {
            const hindiName = hindiNames[name] || name;
            message += `${hindiName}, ${count} की संख्या में मिला है। `;
        });

        message += "धन्यवाद!";

        window.speechSynthesis.cancel();

        const speech = new SpeechSynthesisUtterance(message);
        speech.lang = "hi-IN";
        speech.rate = 0.9;
        speech.pitch = 1;

        window.speechSynthesis.speak(speech);
    });
}

// Stop Speaking
if (stopVoiceButton) {
    stopVoiceButton.addEventListener("click", function () {
        window.speechSynthesis.cancel();
    });
}


// Object Information
const infoButtons = document.querySelectorAll(".info-button");
const objectInfo = document.getElementById("objectInfo");

const objectKnowledge = {
    person: "A person is a human being. AI identifies people using visual patterns.",
    car: "A car is a road vehicle used for transportation.",
    bus: "A bus is a large vehicle designed to carry passengers.",
    bicycle: "A bicycle is a two-wheeled vehicle powered by pedalling.",
    motorcycle: "A motorcycle is a two-wheeled motor vehicle.",
    dog: "A dog is a domesticated animal often kept as a pet.",
    cat: "A cat is a small domesticated animal.",
    bird: "A bird is a feathered animal with a beak and wings.",
    bottle: "A bottle is a container used to store liquids.",
    chair: "A chair is a piece of furniture designed for sitting.",
    laptop: "A laptop is a portable personal computer.",
    cell_phone: "A cell phone is a portable communication device."
};

infoButtons.forEach(button => {
    button.addEventListener("click", function () {
        const name = this.dataset.name.toLowerCase();

        const info = objectKnowledge[name] ||
            "This object was detected by the YOLO AI model. More detailed information is not available yet.";

        objectInfo.innerHTML =
            "<strong>" + name.replaceAll("_", " ").toUpperCase() +
            "</strong><p>" + info + "</p>";
    });
});

// Category Analytics
const analyticsChart =
    document.getElementById("analyticsChart");

if (analyticsChart) {
    try {
        const objects = JSON.parse(
            analyticsChart.dataset.objects || "[]"
        );

        const counts = {};

        objects.forEach(obj => {
            counts[obj.name] = (counts[obj.name] || 0) + 1;
        });

        const entries = Object.entries(counts);

        if (entries.length > 0) {
            analyticsChart.innerHTML = "";

            entries.forEach(([name, count]) => {
                const percentage =
                    (count / objects.length) * 100;

                const row = document.createElement("div");
                row.className = "analytics-row";

                const label = document.createElement("strong");
                label.textContent =
                    name.replaceAll("_", " ") + " (" + count + ")";

                const track = document.createElement("div");
                track.className = "analytics-track";

                const bar = document.createElement("div");
                bar.className = "analytics-fill";
                bar.style.width = percentage + "%";

                track.appendChild(bar);
                row.append(label, track);
                analyticsChart.appendChild(row);
            });
        }
    } catch (error) {
        console.error("Analytics error:", error);
    }
}
