const ctx = document.getElementById("riskChart");

new Chart(ctx, {
    type: "bar",
    data: {
        labels: ["Low", "Moderate", "High"],
        datasets: [{
            label: "Risk Probability",
            data: [30, 45, 25]
        }]
    }
});
