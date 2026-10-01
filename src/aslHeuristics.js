/**
 * An expanded ASL Alphabet Recognizer using heuristics based on MediaPipe Hand Landmarks.
 * Covers A-Z with normalized scale invariance for high accuracy regardless of distance.
 */

export class ASLRecognizer {
  constructor() {
    this.lastLetter = null;
    this.consecutiveFrames = 0;
    this.threshold = 4; // Optimized for speed and stability
    this.wristHistoryX = [];
    this.wristHistoryY = [];
  }

  getDistance(lm1, lm2) {
    return Math.sqrt(
      Math.pow(lm1.x - lm2.x, 2) +
      Math.pow(lm1.y - lm2.y, 2) +
      Math.pow(lm1.z - lm2.z, 2)
    );
  }

  isFingerExtended(tipIdx, dipIdx, pipIdx, mcpIdx, landmarks) {
    // Rotation-invariant check: is the tip further from the wrist than the PIP joint?
    const distTipToWrist = this.getDistance(landmarks[tipIdx], landmarks[0]);
    const distPipToWrist = this.getDistance(landmarks[pipIdx], landmarks[0]);
    return distTipToWrist > distPipToWrist;
  }
  
  isThumbExtended(landmarks, scale) {
    const distToPinky = this.getDistance(landmarks[4], landmarks[17]);
    const distToIndex = this.getDistance(landmarks[4], landmarks[5]);
    return distToPinky > distToIndex * 1.5; 
  }

  isThumbFoldedOver(landmarks, scale) {
    const distToRingMcp = this.getDistance(landmarks[4], landmarks[13]);
    const distToPinkyMcp = this.getDistance(landmarks[4], landmarks[17]);
    // Normalized threshold
    return distToRingMcp < 0.6 * scale || distToPinkyMcp < 0.6 * scale;
  }

  detectLetter(landmarks) {
    if (!landmarks || landmarks.length < 21) {
      this.wristHistoryX = [];
      this.wristHistoryY = [];
      return null;
    }

    // Calculate scale of the hand (Wrist to Middle Finger MCP) to normalize thresholds
    const scale = this.getDistance(landmarks[0], landmarks[9]);
    if (scale === 0) return null;

    // Track wrist position for motion detection (Hello vs Thank You)
    this.wristHistoryX.push(landmarks[0].x);
    this.wristHistoryY.push(landmarks[0].y);
    if (this.wristHistoryX.length > 10) {
      this.wristHistoryX.shift();
      this.wristHistoryY.shift();
    }

    const thumb = this.isThumbExtended(landmarks, scale);
    const thumbFolded = this.isThumbFoldedOver(landmarks, scale);
    
    // Index 8, Middle 12, Ring 16, Pinky 20
    const i = this.isFingerExtended(8, 7, 6, 5, landmarks);
    const m = this.isFingerExtended(12, 11, 10, 9, landmarks);
    const r = this.isFingerExtended(16, 15, 14, 13, landmarks);
    const p = this.isFingerExtended(20, 19, 18, 17, landmarks);

    // Check for Motion Gestures with open hand
    if (i && m && r && p) {
      if (this.wristHistoryX.length === 10) {
        const minX = Math.min(...this.wristHistoryX);
        const maxX = Math.max(...this.wristHistoryX);
        const diffX = maxX - minX;
        
        const startY = this.wristHistoryY[0];
        const endY = this.wristHistoryY[9];
        const diffY = endY - startY; // positive means hand moved down

        // If horizontal motion is greater than vertical and exceeds threshold -> Wave (Hello)
        if (diffX > 0.1 && diffX > Math.abs(diffY)) {
          this.wristHistoryX = []; 
          this.wristHistoryY = [];
          this.lastLetter = "Hello";
          this.consecutiveFrames = this.threshold;
          return "Hello";
        }
        
        // If vertical downward motion is significant -> Thank You
        if (diffY > 0.15 && diffY > diffX) {
          this.wristHistoryX = []; 
          this.wristHistoryY = [];
          this.lastLetter = "Thank you";
          this.consecutiveFrames = this.threshold;
          return "Thank you";
        }
      }
    }

    // Separations (Normalized)
    const sepIM = this.getDistance(landmarks[8], landmarks[12]) / scale;
    const sepMR = this.getDistance(landmarks[12], landmarks[16]) / scale;
    
    const thumbY = landmarks[4].y;
    const indexY = landmarks[8].y;

    let detected = null;

    // Fist based shapes
    if (!i && !m && !r && !p) {
      if (thumb) detected = "A"; 
      else if (thumbFolded) detected = "S"; 
      else if (thumbY < landmarks[6].y) detected = "T"; 
      else detected = "E"; 
      
      if (landmarks[4].x > landmarks[13].x && landmarks[4].x < landmarks[17].x) detected = "M"; 
      if (landmarks[4].x > landmarks[9].x && landmarks[4].x < landmarks[13].x) detected = "N"; 
    }
    // Only Index Out
    else if (i && !m && !r && !p) {
      if (thumb) detected = "L";
      else if (landmarks[8].y > landmarks[5].y) detected = "P"; 
      else if (Math.abs(landmarks[8].x - landmarks[5].x) > 0.4 * scale) detected = "G"; 
      else if (landmarks[8].x < landmarks[7].x) detected = "X"; 
      else detected = "D";
    }
    // Index and Middle Out
    else if (i && m && !r && !p) {
      if (sepIM > 0.2) {
        if (landmarks[8].y > landmarks[5].y) detected = "Q"; 
        else detected = "V";
      } else {
        if ((landmarks[8].x < landmarks[12].x)) detected = "R";
        else detected = "U";
      }
      
      if (Math.abs(landmarks[8].y - landmarks[5].y) < 0.4 * scale && Math.abs(landmarks[12].y - landmarks[9].y) < 0.4 * scale) {
        detected = "H";
      }
      if (thumbY > indexY && thumbY < landmarks[9].y && sepIM > 0.2) detected = "K";
    }
    // Index, Middle, Ring Out
    else if (i && m && r && !p) {
      if (sepIM > 0.15 && sepMR > 0.15) detected = "W";
      else detected = "F"; 
    }
    // Only Pinky Out
    else if (!i && !m && !r && p) {
      if (thumb) detected = "Y";
      else detected = "I";
    }
    
    // C: all fingers curved, thumb out
    else if (!i && !m && !r && !p && thumb && Math.abs(landmarks[8].x - landmarks[4].x) < 0.6 * scale) {
      detected = "C";
    }
    // B: all fingers out, thumb folded
    else if (i && m && r && p) {
      if (!thumb && thumbFolded) detected = "B";
      else detected = null; // Open hand with thumb out is null (waiting for wave or relaxed)
    }

    // Special cases / overrides
    const distThumbIndex = this.getDistance(landmarks[4], landmarks[8]) / scale;
    if (m && r && p && distThumbIndex < 0.3) {
      detected = "F";
    }
    
    const distThumbMiddle = this.getDistance(landmarks[4], landmarks[12]) / scale;
    if (!i && !m && !r && !p && distThumbIndex < 0.4 && distThumbMiddle < 0.4) {
      detected = "O";
    }
    
    // Temporal Smoothing
    if (detected === this.lastLetter && detected !== null) {
      this.consecutiveFrames++;
    } else {
      this.lastLetter = detected;
      this.consecutiveFrames = 0;
    }

    if (this.consecutiveFrames >= this.threshold) {
      return detected;
    }
    return null;
  }
}
