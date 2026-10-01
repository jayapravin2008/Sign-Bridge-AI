import './style.css';
import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { ASLRecognizer } from './aslHeuristics.js';
import { ISLRecognizer } from './islHeuristics.js';

const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loading-overlay');
const captionText = document.getElementById('detected-letter');
const popoutBtn = document.getElementById('popout-btn');
const pipContainer = document.getElementById('pip-container');

const labelAsl = document.getElementById('label-asl');
const labelIsl = document.getElementById('label-isl');
const langSwitch = document.getElementById('lang-switch');

let currentLanguage = 'ASL'; // Default
let detectionThreshold = 5;

langSwitch.addEventListener('change', (e) => {
  if (e.target.checked) {
    currentLanguage = 'ISL';
    labelAsl.classList.remove('active');
    labelIsl.classList.add('active');
  } else {
    currentLanguage = 'ASL';
    labelIsl.classList.remove('active');
    labelAsl.classList.add('active');
  }
  // Clear captions on switch
  captionText.textContent = "Waiting for sign...";
  aslRecognizerLeft.detectLetter(null);
  aslRecognizerRight.detectLetter(null);
  islRecognizer.detectLetter(null, null);
});

let handLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let drawingUtils = null;

// Text-to-Speech State
let lastSpokenText = "";

function speakText(text) {
  if (!text || text === lastSpokenText) return;
  if (text.includes("Waiting") || text.includes("Please use both hands")) return;

  lastSpokenText = text;
  
  window.speechSynthesis.cancel();
  
  // Clean up " | " for dual ASL signs so it sounds natural
  const spokenString = text.replace(/ \| /g, " and ");
  const utterance = new SpeechSynthesisUtterance(spokenString);
  utterance.rate = 1.0; 
  window.speechSynthesis.speak(utterance);
}

// Throttling for performance (FPS cap)
let lastRenderTimeMs = 0;
const targetFPS = 24; 
const frameDelayMs = 1000 / targetFPS;

// ASL (Independent Hands)
const aslRecognizerLeft = new ASLRecognizer();
const aslRecognizerRight = new ASLRecognizer();
// ISL (Joint Hands)
const islRecognizer = new ISLRecognizer();

// Set initial threshold
aslRecognizerLeft.threshold = detectionThreshold;
aslRecognizerRight.threshold = detectionThreshold;
islRecognizer.threshold = detectionThreshold;

// 1. Initialize MediaPipe Hands
async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
  
  loadingOverlay.classList.add('hidden');
  popoutBtn.disabled = false;
  startCamera();
}
createHandLandmarker();

// 2. Start Camera
function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Browser API navigator.mediaDevices.getUserMedia not available');
    return;
  }
  const constraints = {
    video: { facingMode: "user" }
  };
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      webcamRunning = true;
      canvasElement.width = video.videoWidth;
      canvasElement.height = video.videoHeight;
      drawingUtils = new DrawingUtils(canvasCtx);
      predictWebcam();
    });
  }).catch((err) => {
    console.error("Camera Error:", err);
    alert("Could not access the camera: " + err.message);
  });
}

// 3. Render Loop
async function predictWebcam() {
  if (!webcamRunning) return;
  
  // Update canvas size if it changed (e.g. window resize)
  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }

  let startTimeMs = performance.now();
  if (startTimeMs - lastRenderTimeMs >= frameDelayMs) {
    if (lastVideoTime !== video.currentTime) {
      lastRenderTimeMs = startTimeMs;
      lastVideoTime = video.currentTime;
      const results = handLandmarker.detectForVideo(video, startTimeMs);

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      
      if (results.landmarks && results.landmarks.length > 0) {
        
        // Draw skeletons for all detected hands
        for (let i = 0; i < results.landmarks.length; i++) {
          const landmarks = results.landmarks[i];
          const handedness = results.handedness[i][0].categoryName;
          drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
            color: handedness === "Left" ? "#ff00ff" : "#00d2ff",
            lineWidth: 4
          });
          drawingUtils.drawLandmarks(landmarks, { color: "#8a2be2", lineWidth: 2, radius: 4 });
        }

        if (currentLanguage === 'ASL') {
          // ASL ROUTING (Independent)
          let detectedText = [];
          for (let i = 0; i < results.landmarks.length; i++) {
            const landmarks = results.landmarks[i];
            const handedness = results.handedness[i][0].categoryName;
            let letter = null;
            if (handedness === "Left") {
              letter = aslRecognizerLeft.detectLetter(landmarks);
            } else {
              letter = aslRecognizerRight.detectLetter(landmarks);
            }
            if (letter) detectedText.push(letter);
          }
          
          if (detectedText.length > 0) {
            captionText.textContent = detectedText.join(" | ");
            captionText.style.color = '#fff';
            speakText(captionText.textContent);
          } else {
            captionText.textContent = "Waiting for sign...";
            captionText.style.color = 'var(--secondary)';
          }

        } else if (currentLanguage === 'ISL') {
          // ISL ROUTING (Joint Two-Handed)
          let leftLandmarks = null;
          let rightLandmarks = null;
          
          for (let i = 0; i < results.landmarks.length; i++) {
            const handedness = results.handedness[i][0].categoryName;
            if (handedness === "Left") leftLandmarks = results.landmarks[i];
            if (handedness === "Right") rightLandmarks = results.landmarks[i];
          }

          if (leftLandmarks && rightLandmarks) {
            const letter = islRecognizer.detectLetter(leftLandmarks, rightLandmarks);
            if (letter) {
              captionText.textContent = letter;
              captionText.style.color = '#fff';
              speakText(captionText.textContent);
            } else {
              captionText.textContent = "Waiting for ISL sign...";
              captionText.style.color = 'var(--secondary)';
            }
          } else {
            captionText.textContent = "Please use both hands...";
            captionText.style.color = 'var(--secondary)';
            islRecognizer.detectLetter(null, null);
          }
        }
        
      } else {
        captionText.textContent = "Waiting for sign...";
        captionText.style.color = 'var(--secondary)';
        aslRecognizerLeft.detectLetter(null);
        aslRecognizerRight.detectLetter(null);
        islRecognizer.detectLetter(null, null);
      }
      canvasCtx.restore();
    }
  }
  
  // Run loop
  window.requestAnimationFrame(predictWebcam);
}

// 4. Document Picture-in-Picture Setup
popoutBtn.addEventListener('click', async () => {
  if (!('documentPictureInPicture' in window)) {
    alert("Your browser does not support the Document Picture-in-Picture API. Please try Google Chrome or Microsoft Edge (v111+).");
    return;
  }

  try {
    const pipWindow = await window.documentPictureInPicture.requestWindow({
      width: 400,
      height: 300,
    });

    // Copy styles to PiP Window
    const styleSheets = document.styleSheets;
    for (const sheet of styleSheets) {
      try {
        const cssRules = [...sheet.cssRules].map(rule => rule.cssText).join('');
        const style = document.createElement('style');
        style.textContent = cssRules;
        pipWindow.document.head.appendChild(style);
      } catch (e) {
        if (sheet.href) {
          const link = document.createElement('link');
          link.rel = 'stylesheet';
          link.href = sheet.href;
          pipWindow.document.head.appendChild(link);
        }
      }
    }

    // Append our container to the PiP window
    pipWindow.document.body.appendChild(pipContainer);
    pipWindow.document.body.classList.add('in-pip');

    // Handle PiP window close
    pipWindow.addEventListener("pagehide", (event) => {
      const appContainer = document.querySelector('.main-content');
      appContainer.appendChild(pipContainer);
    });
  } catch (error) {
    console.error(error);
    alert("Failed to open PiP window. Check console for details.");
  }
});
