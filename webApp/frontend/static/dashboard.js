(function () {
    const REQUIRED_COLUMNS = [
        "player_id",
        "date",
        "speed_mean",
        "speed_max",
        "speed_std",
        "acc_norm_mean",
        "acc_norm_max",
        "acc_norm_std",
        "gyro_norm_mean",
        "gyro_norm_max",
    ];

    const NUMERIC_COLUMNS = [
        "speed_mean",
        "speed_max",
        "speed_std",
        "acc_norm_mean",
        "acc_norm_max",
        "acc_norm_std",
        "gyro_norm_mean",
        "gyro_norm_max",
    ];

    const form = document.getElementById("predictForm");
    const fileInput = document.getElementById("csvFile");
    const submitButton = document.getElementById("predictBtn");
    const statusNode = document.getElementById("uploadStatus");
    const tableBody = document.getElementById("data-table-body");
    const riskGauge = document.getElementById("riskGauge");
    const riskPercent = document.getElementById("riskPercent");
    const riskLabel = document.getElementById("riskLabel");
    const riskMeta = document.getElementById("riskMeta");
    const sidebar = document.getElementById("sidebar");
    const main = document.getElementById("main");

    const EMPTY_TABLE_ROW = "<tr><td colspan=\"10\" class=\"placeholder-row\">Carica un CSV per vedere l'anteprima.</td></tr>";
    const DATE_REGEX = /^\d{4}-\d{2}-\d{2}$/;

    window.toggleSidebar = function toggleSidebar() {
        sidebar.classList.toggle("hidden");
        main.classList.toggle("full");
    };

    function setStatus(message, kind) {
        statusNode.textContent = message;
        statusNode.className = "status-message";
        statusNode.classList.add(`status-${kind}`);
    }

    function resetPreviewAndRisk() {
        tableBody.innerHTML = EMPTY_TABLE_ROW;
        riskGauge.style.setProperty("--risk", "0");
        riskPercent.textContent = "--%";
        riskLabel.textContent = "In attesa di analisi";
        riskLabel.className = "risk-label risk-neutral";
        riskMeta.textContent = "Carica un CSV e invia i 7 giorni al backend per la predizione.";
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function splitCsvLine(line) {
        const values = [];
        let token = "";
        let inQuotes = false;

        for (let i = 0; i < line.length; i += 1) {
            const char = line[i];

            if (char === "\"") {
                if (inQuotes && line[i + 1] === "\"") {
                    token += "\"";
                    i += 1;
                } else {
                    inQuotes = !inQuotes;
                }
                continue;
            }

            if (char === "," && !inQuotes) {
                values.push(token.trim());
                token = "";
                continue;
            }

            token += char;
        }

        values.push(token.trim());
        return values;
    }

    function parseCsv(text) {
        const lines = text
            .replace(/\uFEFF/g, "")
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line.length > 0);

        if (lines.length < 2) {
            throw new Error("Il file CSV e vuoto o incompleto.");
        }

        const header = splitCsvLine(lines[0]);
        const exactHeader =
            header.length === REQUIRED_COLUMNS.length
            && REQUIRED_COLUMNS.every((col, index) => header[index] === col);

        if (!exactHeader) {
            throw new Error(
                "Header non valido. Usa esattamente: "
                + REQUIRED_COLUMNS.join(","),
            );
        }

        const rows = [];
        for (let i = 1; i < lines.length; i += 1) {
            const columns = splitCsvLine(lines[i]);
            if (columns.length !== REQUIRED_COLUMNS.length) {
                throw new Error(`Riga ${i + 1} non valida: numero colonne errato.`);
            }

            const row = {};
            REQUIRED_COLUMNS.forEach((columnName, index) => {
                row[columnName] = columns[index];
            });

            if (!DATE_REGEX.test(row.date) || Number.isNaN(Date.parse(`${row.date}T00:00:00Z`))) {
                throw new Error(`Riga ${i + 1}: date non valida (${row.date}). Usa YYYY-MM-DD.`);
            }

            NUMERIC_COLUMNS.forEach((columnName) => {
                const parsed = Number(row[columnName]);
                if (!Number.isFinite(parsed)) {
                    throw new Error(`Riga ${i + 1}: ${columnName} non numerico.`);
                }
                row[columnName] = parsed;
            });

            row.__date = new Date(`${row.date}T00:00:00Z`);
            rows.push(row);
        }

        if (rows.length !== 7) {
            throw new Error(`Numero sessioni non valido: trovate ${rows.length}, richieste 7.`);
        }

        rows.sort((a, b) => a.__date - b.__date);
        rows.forEach((row) => delete row.__date);
        return rows;
    }

    function renderPreview(rows) {
        if (!rows.length) {
            tableBody.innerHTML = EMPTY_TABLE_ROW;
            return;
        }

        const bodyHtml = rows.map((row) => `
            <tr>
                <td>${escapeHtml(row.player_id)}</td>
                <td>${escapeHtml(row.date)}</td>
                <td>${row.speed_mean}</td>
                <td>${row.speed_max}</td>
                <td>${row.speed_std}</td>
                <td>${row.acc_norm_mean}</td>
                <td>${row.acc_norm_max}</td>
                <td>${row.acc_norm_std}</td>
                <td>${row.gyro_norm_mean}</td>
                <td>${row.gyro_norm_max}</td>
            </tr>
        `).join("");

        tableBody.innerHTML = bodyHtml;
    }

    function classifyRisk(percent) {
        if (percent < 25) {
            return { text: "Rischio basso", cssClass: "risk-low", details: "Profilo sotto soglia critica." };
        }
        if (percent < 50) {
            return { text: "Rischio medio", cssClass: "risk-medium", details: "Monitorare carico e recupero." };
        }
        if (percent < 75) {
            return { text: "Rischio alto", cssClass: "risk-high", details: "Serve attenzione sul prossimo microciclo." };
        }
        return { text: "Rischio critico", cssClass: "risk-critical", details: "Valutare intervento preventivo immediato." };
    }

    function renderRisk(probability, predictedLabel) {
        const normalized = Math.max(0, Math.min(1, Number(probability)));
        const percent = normalized * 100;
        const risk = classifyRisk(percent);

        riskGauge.style.setProperty("--risk", String(percent));
        riskPercent.textContent = `${percent.toFixed(1)}%`;
        riskLabel.textContent = risk.text;
        riskLabel.className = `risk-label ${risk.cssClass}`;

        if (predictedLabel === null || predictedLabel === undefined) {
            riskMeta.textContent = risk.details;
        } else {
            riskMeta.textContent = `${risk.details} Predicted label: ${predictedLabel}`;
        }
    }

    function toPredictionPayload(rows) {
        return rows.map((row) => ({
            speed_mean: row.speed_mean,
            speed_max: row.speed_max,
            speed_std: row.speed_std,
            acc_norm_mean: row.acc_norm_mean,
            acc_norm_max: row.acc_norm_max,
            acc_norm_std: row.acc_norm_std,
            gyro_norm_mean: row.gyro_norm_mean,
            gyro_norm_max: row.gyro_norm_max,
        }));
    }

    async function submitPrediction(event) {
        event.preventDefault();

        const selectedFile = fileInput.files && fileInput.files[0];
        if (!selectedFile) {
            setStatus("Seleziona prima un file CSV.", "error");
            return;
        }

        submitButton.disabled = true;
        setStatus("Validazione CSV in corso...", "info");

        try {
            const text = await selectedFile.text();
            const rows = parseCsv(text);
            renderPreview(rows);

            setStatus("Invio dei 7 giorni al backend...", "info");

            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ last_7_days: toPredictionPayload(rows) }),
            });

            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.error || "Errore backend durante la predizione.");
            }

            if (!Number.isFinite(Number(payload.injury_probability))) {
                throw new Error("Risposta backend non valida: injury_probability mancante.");
            }

            renderRisk(payload.injury_probability, payload.predicted_label);
            setStatus("Predizione completata con successo.", "success");
        } catch (error) {
            resetPreviewAndRisk();
            setStatus(error.message || "Errore non gestito.", "error");
        } finally {
            submitButton.disabled = false;
        }
    }

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files[0]) {
            setStatus(`File selezionato: ${fileInput.files[0].name}`, "info");
        } else {
            setStatus("Nessun file selezionato.", "info");
        }
    });

    form.addEventListener("submit", submitPrediction);
    resetPreviewAndRisk();
}());
