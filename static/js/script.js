document.addEventListener("DOMContentLoaded", function () {
    console.log("🟢 JS is running");

    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("file");
    const modelSelect = document.getElementById("model");

    form.addEventListener("submit", function (event) {
        const file = fileInput.files[0];
        const model = modelSelect.value;

        if (!file || !model) {
            alert("❌ Please select both a file and a model.");
            event.preventDefault(); // Empêche l'envoi du formulaire
        } else {
            console.log("✅ File and model selected");
        }
    });
});
