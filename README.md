# AC215 Project

Members: Kaylee Vo, Chloe Seo, Ben Nguyen, Patrick Nguyen

## Project Overview

### Phase 1: Blood Pressure Prediction
In phase 1 of the project, we build two models to predict the blood pressure of patients, an online model (dynamic dataset) and an offline model (static dataset). Blood pressure consists of systolic pressure and diastolic pressure (mmHg). We build a univariate, sequence-to-sequence LSTM model to forecast each blood pressure reading. The aim of the project is to use predictions for earlier clinician and nurse intervention. We simulate a real-time data feed by gradually adding records into the database on a fixed time interval. This creates an artificial time lag. To achieve real-time blood pressure readings on the dashboard, we will nowcast.

#### Data
For our data, we use individual-level blood pressure data from VitalDB [1]. 

#### Architecture
TBD

### Phase 2: Digital Twin
In phase 2, we build a digital twin to anticipate a patient's reaction to treatment. We use clinical data, which gives patient-level covariates and the treatment effect of a given drug on blood pressure. 


### Sources

1. VitalDB. (n.d.). Dataset : VitalDB. https://vitaldb.net/dataset/