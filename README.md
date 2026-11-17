# Quality-Aware Deep Learning for Robust Non-Contact Blood Pressure Estimation Using iPPG

An academic project from the Department of Computer Science and Engineering at Bapatla Women's Engineering College. The project investigates non-contact physiological sensing through imaging photoplethysmography (iPPG), which analyses small changes in skin colour in camera video.

The current application combines face tracking, camera-derived pulse-signal analysis, and a simulated blood-pressure interface. All camera processing runs in the browser.

[Source code](https://github.com/mullamahi/iPPG-Bp) · [Run locally](#run-locally) · [Current implementation](#current-implementation) · [Academic credits](#academic-credits)

> **Prototype status:** The displayed blood pressure, heart rate, and frequency spectrum are simulated. The recovered signal and quality checks use camera data. No trained blood-pressure estimation model or external blood-pressure service is connected. This demonstration is not a medical device and must not guide diagnosis or treatment.

## Current implementation

The project title describes the research objective. The repository currently implements a browser prototype, rather than an end-to-end deep-learning blood-pressure estimator. MediaPipe supplies the face-tracking model; pulse extraction uses signal-processing methods.

| Component               | Current behaviour                                                                          | Data source                                             |
| ----------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| Face tracking           | Tracks one face and overlays forehead and cheek regions                                    | Live camera frames                                      |
| Signal                  | Plots the recovered pulse waveform                                                         | Camera-derived RGB samples                              |
| Quality checks          | Reports face visibility, lighting, motion, and signal quality                              | Face geometry and sampled signals                       |
| Pulse signal confidence | Scores the quality of the recovered pulse signal                                           | Signal analysis; not blood-pressure accuracy            |
| Heart rate              | Displays a slowly varying simulated resting pulse after the signal passes the display gate | Synthetic values                                        |
| Blood pressure          | Displays simulated systolic and diastolic values in mmHg                                   | Synthetic values independent of pulse and camera pixels |
| Spectrum                | Plots a generated peak aligned with the displayed simulated pulse                          | Synthetic spectrum                                      |
| Blood pressure details  | Freezes the displayed pair and adds a simulated category after a valid **Stop**            | Frozen synthetic values                                 |
| Session summary         | Reports average pulse, pulse range, and camera frame rate                                  | Simulated pulse statistics and camera timing            |

The signal-processing module computes a camera-derived heart-rate estimate internally. The interface currently displays a synthetic pulse instead of that estimate. A high confidence score therefore does not validate the displayed heart rate or blood pressure.

## Interface

The page uses pearl white backgrounds, cherry red accents, and green signal and spectrum lines. Responsive layouts adapt the cards to narrower screens.

The interface contains:

- A live camera panel with face-region overlays, status messages, and positioning guidance.
- **Start measurement**, **Stop**, **Reset**, and camera full-screen controls.
- A blood-pressure card with simulated readings, pulse signal confidence, elapsed time, and collection progress.
- Four quality indicators for **Face**, **Lighting**, **Motion**, and **Signal**.
- Separate **Signal** and **Spectrum** charts.
- A final blood-pressure details panel and session statistics.
- Academic credits for the institution, project team, and guide.

The page includes a skip link, keyboard focus indicators, labelled controls, live status announcements, and reduced-motion styles. These provisions do not constitute a completed accessibility audit.

## Run locally

### Requirements

- Node.js 20 or later and npm. The continuous integration workflow uses Node.js 22.
- A browser with JavaScript, WebAssembly, and camera access enabled.
- A working camera.
- Internet access to download the static server, MediaPipe runtime, and face-tracking model when needed.

### Start the application

Run these commands in a terminal:

```bash
git clone https://github.com/mullamahi/iPPG-Bp.git
cd iPPG-Bp
npm start
```

Open the local URL printed by the server. If you already have the project, run `npm start` in its directory.

The start script invokes `npx --yes serve .`. There is no application build step, backend service, database, or account setup.

Camera access requires a secure context. Use `localhost` for local development and HTTPS for a hosted deployment. An HTTP address on another device's local network can prevent camera access.

## Use the demonstration

1. Place the camera on a stable surface with steady light in front of your face.
2. Select **Start measurement** and allow camera access.
3. Keep one face visible, look forward, and leave your forehead and cheeks unobstructed.
4. Follow the camera guidance and watch the quality indicators as the signal accumulates.
5. Wait for the simulated readings to appear. The display requires at least 12 seconds of sampled signal, a recently tracked face, and sufficient signal confidence.
6. Select **Stop** to finish. A valid session preserves the displayed readings and fills the blood-pressure details panel.
7. Select **Reset** to clear the session, or select **Start measurement** to begin another capture.

Reaching 12 seconds does not stop the camera automatically. Capture continues until you select **Stop** or the session is interrupted. If the signal remains weak, the readings can remain unavailable beyond 12 seconds.

An early stop or insufficient signal does not produce a completed result. Hiding the tab stops an active capture without finalising it. Stopping a capture releases the camera tracks.

## Processing pipeline

Camera analysis and simulated display values follow separate paths.

1. **Capture frames.** The app requests a front-facing camera without audio. It uses video-frame callbacks where available, with an animation-frame fallback.
2. **Track the face.** MediaPipe Face Landmarker identifies facial landmarks. The app tries GPU execution first and falls back to CPU execution if initialisation fails.
3. **Sample skin regions.** Face-relative polygons cover the forehead and both cheeks. The app extracts regional RGB averages and checks luminance, clipping, and pixel coverage.
4. **Resample the signal.** The signal module converts irregularly timed samples into uniformly spaced colour series.
5. **Extract pulse signals.** Plane-Orthogonal-to-Skin (POS), chrominance-based (CHROM), and green-channel methods produce candidate signals for each region.
6. **Analyse frequency content.** Windowed spectra and a weighted ensemble produce an internal pulse estimate. The application uses a 12-second analysis window and searches between 45 and 200 BPM.
7. **Assess quality.** Confidence combines spectral strength, competing peaks, agreement between methods, frame timing, lighting, motion, pose, face quality, and signal duration.
8. **Update the interface.** The recovered waveform and confidence reflect the camera signal. Once the display gate passes, separate generators supply the simulated pulse, blood pressure, and spectrum.

The blood-pressure category describes the final synthetic pair. It does not establish a physiological relationship between the camera signal and blood pressure.

## Technology and source layout

| Layer                  | Implementation                                            |
| ---------------------- | --------------------------------------------------------- |
| Interface              | HTML, CSS, and JavaScript ES modules                      |
| Camera                 | Browser MediaDevices API                                  |
| Face tracking          | MediaPipe Tasks Vision, pinned to `1.0.1` in `src/app.js` |
| Face model             | Google-hosted Face Landmarker model                       |
| Signal processing      | Local JavaScript implementation in `src/ippg.js`          |
| Charts                 | Canvas 2D API                                             |
| Local server           | `serve`, invoked through npm                              |
| Tests                  | Built-in Node.js test runner                              |
| Continuous integration | GitHub Actions                                            |

| Path                                                       | Responsibility                                                                   |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [`index.html`](index.html)                                 | Page structure, visible controls, labels, and academic credits                   |
| [`styles.css`](styles.css)                                 | Colour tokens, component styles, responsive layouts, and motion preferences      |
| [`src/app.js`](src/app.js)                                 | Camera lifecycle, face tracking, skin sampling, display simulation, and UI state |
| [`src/ippg.js`](src/ippg.js)                               | Sample buffering, resampling, pulse extraction, spectra, and confidence scoring  |
| [`src/charts.js`](src/charts.js)                           | Signal and spectrum rendering                                                    |
| [`tests/ippg.test.mjs`](tests/ippg.test.mjs)               | Synthetic-signal tests for the analysis module                                   |
| [`tests/capture.test.mjs`](tests/capture.test.mjs)         | Capture-state and simulated-display tests with browser API stubs                 |
| [`.github/workflows/test.yml`](.github/workflows/test.yml) | Test workflow for pushes to `main` and pull requests                             |

`IppgBuffer` provides a `toCsv()` method for developers. The current page does not expose a CSV download control, camera selector, or analysis-settings form.

## Tests and validation

Run the existing test suite:

```bash
npm test
```

The tests cover signal resampling, spectral peak recovery, POS extraction, confidence penalties, capture continuation, stop and reset behaviour, simulated readings, and spectrum synchronisation. They also check that the final category uses the same frozen blood-pressure pair shown in the interface.

These are deterministic software tests with synthetic input and browser API stubs. They do not establish clinical accuracy, validate physical camera hardware, or replace testing in real browsers.

## Privacy and network access

The application processes camera frames locally. Its source contains no frame-upload, account, analytics, or server-side inference integration. Audio capture is disabled.

Local processing still requires external assets. When you start the tracker, the browser requests the MediaPipe JavaScript and WebAssembly runtime from jsDelivr and the face model from Google-hosted storage. A hosted deployment also requests the page assets from its hosting provider.

Session samples remain in application memory. The app does not write them to browser storage. Reloading the page clears the session. Fully offline operation is not configured; cached assets alone are not a guaranteed offline installation.

## Troubleshooting

| Symptom                            | Action                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Camera permission is denied        | Allow camera access for the site in browser settings, then start again.                              |
| Camera access is unavailable       | Use `localhost` or HTTPS and check that the browser exposes camera access.                           |
| Camera fails to start              | Close other applications using the camera, check the device connection, and retry.                   |
| Face tracker fails to load         | Check the network connection and access to jsDelivr and Google-hosted model storage, then reload.    |
| Readings remain blank              | Keep one face visible, improve front lighting, reduce movement, and allow more signal to accumulate. |
| Values disappear during capture    | Check the quality indicators. Readings are withheld when the display conditions are no longer met.   |
| Capture stops after switching tabs | Keep the page visible during capture. Start a new session after returning.                           |
| Full-screen mode is unavailable    | Continue with the embedded camera view. Full-screen availability depends on the browser.             |

## Research scope

The implemented prototype supports work on face tracking, regional pulse extraction, signal-quality assessment, and the measurement interface.

The blood-pressure research objective still requires a trained estimator, appropriate reference measurements, an evaluation protocol, and documented validation results. This repository does not provide evidence of clinical performance or accuracy across devices, skin tones, lighting conditions, or patient populations.

## Academic credits

| Detail                 | Information                                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| Institution            | Bapatla Women's Engineering College                                     |
| Location               | Bapatla 522101, Andhra Pradesh                                          |
| Department             | Computer Science and Engineering (CSE)                                  |
| Project guide          | Mrs. G. Venkateswari, M.Tech (Ph.D., as listed), HOD, Department of CSE |


### Project team

| Name                | Student ID | Role        |
| ------------------- | ---------- | ----------- |
| P. Dhanalakshmi     | Y23CS2676  | Team Lead   |
| SK. Asin            | Y23CS2702  | Team Member |
| M. Mahisha Sulthana | Y23CS2669  | Team Member |


