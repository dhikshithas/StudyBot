document.getElementById("openApp").addEventListener("click", function () {
    chrome.tabs.create({ url: "http://localhost:8501" });
});
