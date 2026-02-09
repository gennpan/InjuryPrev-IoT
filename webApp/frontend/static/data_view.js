(function () {
    const sidebar = document.getElementById("sidebar");
    const main = document.getElementById("main");
    const toggleButton = document.getElementById("toggleFeatures");
    const featuresContainer = document.getElementById("featuresContainer");

    window.toggleSidebar = function toggleSidebar() {
        sidebar.classList.toggle("hidden");
        main.classList.toggle("full");
    };

    toggleButton.addEventListener("click", () => {
        featuresContainer.classList.toggle("hidden");
    });
}());
