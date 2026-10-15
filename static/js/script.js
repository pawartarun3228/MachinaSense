
"use strict";

/* =========================================================
   MachinaSense Dashboard
   Health, model metrics, sensor chart, alerts and CSV upload
   ========================================================= */

const API = {
    health: "/api/health",
    metrics: "/api/metrics",
    alerts: "/api/alerts",
    sensors: "/api/sensor-data?limit=1500",
    upload: "/api/upload"
};

let sensorChart = null;
let sensorRecords = [];
let currentAlerts = [];

const $ = (id) => document.getElementById(id);

function setText(id, value) {
    const element = $(id);
    if (element) element.textContent = value;
}

function formatNumber(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function formatPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";

    // Metrics such as accuracy are usually stored from 0 to 1.
    return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
}

async function getJSON(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.error || `Request failed (${response.status})`);
    }

    return data;
}

/* -------------------- System health -------------------- */

async function loadHealth() {
    try {
        const data = await getJSON(API.health);
        const online =
            data.status === "online" &&
            data.model_ready !== false;

        setText("connectionStatus", online ? "Connected" : "Offline");
        setText("systemStatus", online ? "Operational" : "Unavailable");
    } catch (error) {
        setText("connectionStatus", "Disconnected");
        setText("systemStatus", "Unavailable");
        console.error("Health check failed:", error);
    }
}

/* -------------------- Model evaluation -------------------- */


async function loadMetrics() {

    try {
        const data = await getJSON(API.metrics);

        // Supports either a direct metrics object or a nested one.
        const metrics = data.metrics || data;


        setText("recordsAnalyzed",
            metrics.test_rows ?? metrics.test_samples ?? metrics.records_analyzed ?? "—");


        setText("modelF1", formatPercent(metrics.f1));
        setText("modelPrecision", formatPercent(metrics.precision));
        setText("evalAccuracy", formatPercent(metrics.accuracy));
        setText("evalRecall", formatPercent(metrics.recall));
        setText("evalAP", formatPercent(
            metrics.average_precision ?? metrics.averagePrecision
        ));

        setText("evalFalsePositives",
            metrics.false_positives ?? "—");

        setText("evalThreshold",
            metrics.threshold !== undefined
                ? formatNumber(metrics.threshold, 2)
                : "—");

        const matrix = metrics.confusion_matrix;

        if (Array.isArray(matrix) && matrix.length >= 2) {
            setText("tn", matrix[0]?.[0] ?? "—");
            setText("fp", matrix[0]?.[1] ?? "—");
            setText("fn", matrix[1]?.[0] ?? "—");
            setText("tp", matrix[1]?.[1] ?? "—");
        }
    } catch (error) {
        console.error("Could not load model metrics:", error);
    }
}

/* -------------------- Maintenance alerts -------------------- */

function renderAlerts(alerts) {
    const table = $("alertsTable");
    if (!table) return;

    const tbody = table.tagName === "TBODY"
        ? table
        : table.querySelector("tbody");

    if (!tbody) return;

    tbody.replaceChildren();

    if (!alerts.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 5;
        cell.textContent = "No maintenance alerts found.";
        row.appendChild(cell);
        tbody.appendChild(row);
        return;
    }

    alerts.forEach((alert, index) => {
        const row = document.createElement("tr");

        const probability = Number(
            alert.failure_probability ??
            alert.probability ??
            alert.risk ??
            0
        );

        const probabilityText =
            `${(probability <= 1 ? probability * 100 : probability).toFixed(2)}%`;

        const values = [
            alert.UDI ?? alert.udi ?? alert.id ?? `Record ${index + 1}`,
            probabilityText,

            alert.severity ?? (
                probability >= 0.80 ? "Critical" :
                    probability >= 0.60 ? "High" :
                        probability >= 0.35 ? "Medium" : "Low"
            ),
            alert.predicted_failure
                ? (
                    probability >= 0.80 ? "Prioritize prompt maintenance assessment" :
                        probability >= 0.60 ? "Schedule a priority maintenance inspection" :
                            "Schedule a preventive inspection"
                )
                : "Continue routine monitoring"

        ];

        values.forEach((value) => {
            const cell = document.createElement("td");
            cell.textContent = String(value);
            row.appendChild(cell);
        });

        tbody.appendChild(row);
    });
}

async function loadAlerts() {
    try {
        const data = await getJSON(API.alerts);

        currentAlerts = Array.isArray(data)
            ? data
            : (data.alerts ?? data.results ?? []);

        renderAlerts(currentAlerts);
        setText("failureAlerts", currentAlerts.length);
    } catch (error) {
        console.error("Could not load maintenance alerts:", error);
    }
}

/* -------------------- Sensor monitoring chart -------------------- */

function getSensorRows(data) {
    if (Array.isArray(data)) return data;
    if (Array.isArray(data.records)) return data.records;
    if (Array.isArray(data.data)) return data.data;
    if (Array.isArray(data.results)) return data.results;
    return [];
}

async function loadSensorData() {
    try {
        const data = await getJSON(API.sensors);
        sensorRecords = getSensorRows(data);

        renderSensorChart();
    } catch (error) {
        console.error("Could not load sensor data:", error);

        const message = $("chartMessage");
        if (message) {
            message.textContent =
                "Sensor data could not be loaded. Check the backend endpoint.";
        }
    }
}

function renderSensorChart() {
    const canvas = $("sensorChart");
    if (!canvas) return;

    if (typeof Chart === "undefined") {
        const message = $("chartMessage");
        if (message) message.textContent = "Chart.js is not loaded.";
        return;
    }

    const selector = $("sensorSelector");
    const selectedSensor = selector?.value || "Air_temperature";

    const validRecords = sensorRecords.filter((record) =>
        record[selectedSensor] !== undefined &&
        Number.isFinite(Number(record[selectedSensor]))
    );

    if (!validRecords.length) {
        const message = $("chartMessage");
        if (message) {
            message.textContent = "No data available for this sensor.";
        }

        if (sensorChart) {
            sensorChart.destroy();
            sensorChart = null;
        }
        return;
    }

    const message = $("chartMessage");
    if (message) {
        message.textContent =
            `Showing ${validRecords.length} sensor records.`;
    }

    if (sensorChart) {
        sensorChart.destroy();
    }

    sensorChart = new Chart(canvas, {
        type: "line",
        data: {
            labels: validRecords.map((record, index) =>
                record.UDI ?? record.id ?? index + 1
            ),
            datasets: [{
                label: selectedSensor.replaceAll("_", " "),
                data: validRecords.map((record) =>
                    Number(record[selectedSensor])
                ),
                borderColor: "#48b9ff",
                backgroundColor: "rgba(72, 185, 255, 0.12)",
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: "index"
            },
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Record"
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: selectedSensor.replaceAll("_", " ")
                    }
                }
            }
        }
    });
}

/* -------------------- CSV upload and results -------------------- */

function renderUploadResults(results) {
    const container = $("uploadResults");
    if (!container) return;

    container.replaceChildren();

    const heading = document.createElement("h3");
    heading.textContent = "Uploaded CSV Predictions";
    container.appendChild(heading);

    if (!results.length) {
        const message = document.createElement("p");
        message.textContent = "No prediction results were returned.";
        container.appendChild(message);
        return;
    }

    const summary = document.createElement("p");
    const failures = results.filter((item) =>
        Number(item.predicted_failure) === 1
    ).length;

    summary.textContent =
        `${results.length} rows analyzed · ${failures} predicted failure(s)`;
    container.appendChild(summary);

    const wrapper = document.createElement("div");
    wrapper.style.overflowX = "auto";

    const table = document.createElement("table");
    table.style.width = "100%";

    const headers = [
        "Row",
        "Machine Type",
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear",
        "Failure Probability",
        "Prediction"
    ];

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    headers.forEach((label) => {
        const th = document.createElement("th");
        th.textContent = label;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    results.forEach((item, index) => {
        const row = document.createElement("tr");

        const probability = Number(item.failure_probability ?? 0);
        const probabilityPercent =
            (probability <= 1 ? probability * 100 : probability).toFixed(2);

        const predictedFailure =
            Number(item.predicted_failure) === 1;

        const values = [
            index + 1,
            item.Type ?? "—",
            item.Air_temperature ?? "—",
            item.Process_temperature ?? "—",
            item.Rotational_speed ?? "—",
            item.Torque ?? "—",
            item.Tool_wear ?? "—",
            `${probabilityPercent}%`,
            predictedFailure ? "Failure predicted" : "Normal"
        ];

        values.forEach((value, cellIndex) => {
            const td = document.createElement("td");
            td.textContent = String(value);

            if (cellIndex === 8) {
                td.style.fontWeight = "600";
                td.style.color = predictedFailure
                    ? "#ff6b6b"
                    : "#45d483";
            }

            row.appendChild(td);
        });

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);

    // Show a short model-use warning when supplied by the API.
    const note = document.createElement("p");
    note.textContent = "Model estimates only; predictions do not confirm mechanical faults.";
    note.style.marginTop = "12px";
    container.appendChild(note);
}

function setUploadStatus(message, isError = false) {
    const status = $("uploadStatus");
    const container = $("uploadResults");

    if (status) {
        status.textContent = message;
        status.style.color = isError ? "#ff6b6b" : "#45d483";
    }

    // Keep the error in one place only.
    if (container && isError) {
        container.replaceChildren();
    }
}

function setupCSVUpload() {
    const form = $("uploadForm");
    const fileInput = $("csvFile");

    if (!form || !fileInput) {
        console.warn(
            "CSV upload controls not found. Expected #uploadForm and #csvFile."
        );
        return;
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const file = fileInput.files?.[0];

        if (!file) {
            setUploadStatus("Please choose a CSV file first.", true);
            return;
        }

        if (!file.name.toLowerCase().endsWith(".csv")) {
            setUploadStatus("Please select a .csv file.", true);
            return;
        }

        const submitButton = form.querySelector(
            'button[type="submit"], input[type="submit"]'
        );

        if (submitButton) submitButton.disabled = true;

        setUploadStatus(`Analyzing ${file.name}...`);

        const container = $("uploadResults");
        if (container) container.replaceChildren();

        try {
            const formData = new FormData();
            formData.append("file", file);

            const data = await getJSON(API.upload, {
                method: "POST",
                body: formData
            });

            const results = Array.isArray(data.results)
                ? data.results
                : [];

            renderUploadResults(results);

            const rowCount = data.rows_analyzed ?? results.length;
            const alertCount = data.alerts ?? results.filter(
                (item) => Number(item.predicted_failure) === 1
            ).length;

            setUploadStatus(
                `Analysis complete: ${rowCount} rows analyzed, ${alertCount} predicted failure alert(s).`
            );
        } catch (error) {
            console.error("CSV upload failed:", error);
            setUploadStatus(error.message || "CSV analysis failed.", true);

            {
                if (container) container.replaceChildren();

            }
        } finally {
            if (submitButton) submitButton.disabled = false;
        }
    });
}

/* -------------------- Dashboard initialization -------------------- */

async function refreshDashboard() {
    await Promise.allSettled([
        loadHealth(),
        loadMetrics(),
        loadAlerts(),
        loadSensorData()
    ]);
}

document.addEventListener("DOMContentLoaded", () => {
    setupCSVUpload();

    const selector = $("sensorSelector");
    if (selector) {
        selector.addEventListener("change", renderSensorChart);
    }

    const refreshButton = $("refreshButton");
    if (refreshButton) {
        refreshButton.addEventListener("click", refreshDashboard);
    }

    refreshDashboard();
});
