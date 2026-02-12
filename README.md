# InjuryPrev

**InjuryPrev** è una web application a scopo didattico progettata per il contesto sportivo, in particolare per il calcio.  
L’obiettivo del progetto è stimare in modo predittivo il rischio di infortunio di un atleta tramite una rete neurale artificiale.

Alla base dell’applicazione è stata implementata una **Fully Connected Neural Network (FCNN)**, addestrata sul dataset *Soccermon*, selezionato per la sua completezza rispetto alle metriche richieste dal modello.

---

## Funzionalità principali

### Input

L’applicazione riceve in input un file .csv contenente i dati relativi alle ultime 7 giornate di allenamento di un calciatore.

Il file deve contenere una riga per ciascun giorno di allenamento e rispettare il seguente formato:

```bash
player_id,date,speed_mean,speed_max,speed_std,acc_norm_mean,acc_norm_max,acc_norm_std,gyro_norm_mean,gyro_norm_max
```

#### Struttura dei campi di input

| Campo              | Tipo        | Descrizione |
|--------------------|------------|-------------|
| player_id          | String(UUID-like)    | Identificativo univoco del calciatore a cui si riferiscono i dati di allenamento. |
| date               | Date((YYYY-MM-DD))       | Data della sessione di allenamento (formato consigliato: YYYY-MM-DD). |
| speed_mean         | Float      | Velocità media registrata durante la sessione di allenamento (m/s). |
| speed_max          | Float      | Velocità massima raggiunta durante la sessione di allenamento (m/s). |
| speed_std          | Float      | Deviazione standard della velocità, indicativa della variabilità dell’intensità di corsa. |
| acc_norm_mean      | Float      | Valore medio della norma dell’accelerazione, misura dell’intensità complessiva dei movimenti. |
| acc_norm_max       | Float      | Valore massimo della norma dell’accelerazione registrato durante la sessione. |
| acc_norm_std       | Float      | Deviazione standard della norma dell’accelerazione, utile per valutare la variabilità dei carichi. |
| gyro_norm_mean     | Float      | Valore medio della norma del giroscopio, indicativo dei movimenti rotazionali del corpo. |
| gyro_norm_max      | Float      | Valore massimo della norma del giroscopio registrato durante la sessione. |

### Output

Il sistema elabora i dati forniti in input e restituisce una stima del rischio di infortunio, calcolata sulla base delle metriche contenute nel file caricato.

---

## Dataset

Le metriche presenti nel dataset sono state raccolte tramite il sistema di tracciamento **STATSports APEX GNSS**, approvato dalla FIFA.

I giocatori erano equipaggiati con dispositivi comprendenti:

- unità multi-GNSS a 10 Hz (GPS, GLONASS, Galileo, BeiDou);
- accelerometro triassiale (952 Hz);
- magnetometro (10 Hz);
- giroscopio triassiale (952 Hz).

Tutti i sensori sono integrati nel gilet STATSports, consentendo un monitoraggio completo delle prestazioni atletiche.

---

## Stack Tecnologico

### Frontend 
- HTML5
- CSS3
- JavaScript Vanilla (ES6+)
### Backend
- Python 3.14+
- Pytorch 2.10.0+
- Flask 3.1.2+

### IDE
- WebStorm

---

## Setup del progetto

Il progetto utilizza Python 3.14 all’interno di un ambiente virtuale.

### Guida all’installazione

1. Dopo aver clonato (oppure scaricato ed aperto in webStorm) la repository, aprire il terminale in WebStorm e spostarsi nella cartella del progetto:

```bash
cd InjuryPrev
```

2. Creare l’ambiente virtuale:

```bash
python -m venv .venv
```

3. Attivare l’ambiente:

**Windows (PowerShell)**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

4. Installare le dipendenze:

```bash
pip install -r requirements.txt
```

5. Avviare l’applicazione:

```bash
python run.py
```

A questo punto la webApp sarà accessibile in locale.

---

## Struttura del progetto

Il codice è organizzato in modo modulare per migliorarne manutenibilità ed estendibilità:

```
InjuryPrev/
│
├── api/        # Endpoint backend
├── dataset/    # Dataset utilizzato
├── frontend/   # Interfaccia utente
├── models/     # Modelli di machine learning
├── outputs/    # Risultati delle predizioni
└── scripts/    # Script di supporto
```

---

## Note

Questo progetto è stato sviluppato esclusivamente a fini didattici.
