/**
 * Indian Sign Language (ISL) Recognizer Prototype
 * ISL relies heavily on two-handed gestures. This engine joint-processes both hands.
 */

export class ISLRecognizer {
  constructor() {
    this.lastLetter = null;
    this.consecutiveFrames = 0;
    this.threshold = 4;
  }

  getDistance(lm1, lm2) {
    if (!lm1 || !lm2) return Infinity;
    return Math.sqrt(
      Math.pow(lm1.x - lm2.x, 2) +
      Math.pow(lm1.y - lm2.y, 2) +
      Math.pow(lm1.z - lm2.z, 2)
    );
  }

  isFingerExtended(tipIdx, pipIdx, landmarks) {
    // Rotation-invariant check: is the tip further from the wrist than the PIP joint?
    // This allows detection even when hands are held horizontally or upside down.
    const distTipToWrist = this.getDistance(landmarks[tipIdx], landmarks[0]);
    const distPipToWrist = this.getDistance(landmarks[pipIdx], landmarks[0]);
    return distTipToWrist > distPipToWrist;
  }

  isFist(landmarks) {
    const i = this.isFingerExtended(8, 6, landmarks);
    const m = this.isFingerExtended(12, 10, landmarks);
    const r = this.isFingerExtended(16, 14, landmarks);
    const p = this.isFingerExtended(20, 18, landmarks);
    return !i && !m && !r && !p;
  }

  isOpenHand(landmarks) {
    const i = this.isFingerExtended(8, 6, landmarks);
    const m = this.isFingerExtended(12, 10, landmarks);
    const r = this.isFingerExtended(16, 14, landmarks);
    const p = this.isFingerExtended(20, 18, landmarks);
    return i && m && r && p;
  }

  detectLetter(leftLandmarks, rightLandmarks) {
    if (!leftLandmarks || !rightLandmarks) return null;

    const scale = this.getDistance(rightLandmarks[0], rightLandmarks[9]);
    if (scale === 0) return null;

    // Left Hand Status
    const lIndexUp = this.isFingerExtended(8, 6, leftLandmarks);
    const lMiddleUp = this.isFingerExtended(12, 10, leftLandmarks);
    const lRingUp = this.isFingerExtended(16, 14, leftLandmarks);
    const lPinkyUp = this.isFingerExtended(20, 18, leftLandmarks);
    const lOpen = lIndexUp && lMiddleUp && lRingUp && lPinkyUp;
    const lFist = !lIndexUp && !lMiddleUp && !lRingUp && !lPinkyUp;

    // Right Hand Status
    const rIndexUp = this.isFingerExtended(8, 6, rightLandmarks);
    const rMiddleUp = this.isFingerExtended(12, 10, rightLandmarks);
    const rRingUp = this.isFingerExtended(16, 14, rightLandmarks);
    const rPinkyUp = this.isFingerExtended(20, 18, rightLandmarks);
    const rOpen = rIndexUp && rMiddleUp && rRingUp && rPinkyUp;
    const rFist = !rIndexUp && !rMiddleUp && !rRingUp && !rPinkyUp;

    let detected = null;

    // VOWELS (ISL/BSL Vowels use Right Index touching Left fingertips)
    // Left hand is open, right hand has only index extended
    if (lOpen && rIndexUp && !rMiddleUp && !rRingUp && !rPinkyUp) {
      const dToA = this.getDistance(rightLandmarks[8], leftLandmarks[4]) / scale;
      const dToE = this.getDistance(rightLandmarks[8], leftLandmarks[8]) / scale;
      const dToI = this.getDistance(rightLandmarks[8], leftLandmarks[12]) / scale;
      const dToO = this.getDistance(rightLandmarks[8], leftLandmarks[16]) / scale;
      const dToU = this.getDistance(rightLandmarks[8], leftLandmarks[20]) / scale;
      
      const minDist = Math.min(dToA, dToE, dToI, dToO, dToU);
      if (minDist < 0.25) {
        if (minDist === dToA) detected = "A";
        else if (minDist === dToE) detected = "E";
        else if (minDist === dToI) detected = "I";
        else if (minDist === dToO) detected = "O";
        else if (minDist === dToU) detected = "U";
      }
    }

    if (!detected) {
      // B: Both hands open, index and thumb touching to form circles
      const lThumbIndexClose = this.getDistance(leftLandmarks[4], leftLandmarks[8]) / scale < 0.2;
      const rThumbIndexClose = this.getDistance(rightLandmarks[4], rightLandmarks[8]) / scale < 0.2;
      if (lThumbIndexClose && rThumbIndexClose) {
        const handsClose = this.getDistance(leftLandmarks[8], rightLandmarks[8]) / scale < 0.3;
        if (handsClose) detected = "B";
      }

      // C: Right hand forms a 'C' shape (index and thumb curved, distance moderate).
      else if (!rMiddleUp && !lIndexUp) {
        const cDist = this.getDistance(rightLandmarks[8], rightLandmarks[4]) / scale;
        if (cDist > 0.3 && cDist < 0.7) {
          detected = "C";
        }
      }

      // D: Right index up, left thumb/index circle touches right index.
      else if (rIndexUp && !rMiddleUp && !lMiddleUp) {
        if (lThumbIndexClose && (this.getDistance(leftLandmarks[8], rightLandmarks[8]) / scale < 0.3)) {
          detected = "D";
        }
      }

      // F: Right index/middle crossed over left index/middle
      else if (rIndexUp && rMiddleUp && lIndexUp && lMiddleUp && !rRingUp && !lRingUp) {
        const crossingDist = this.getDistance(rightLandmarks[10], leftLandmarks[10]) / scale;
        if (crossingDist < 0.4) detected = "F";
      }

      // G: Both fists, right on top of left
      else if (lFist && rFist) {
        const yDiff = rightLandmarks[0].y - leftLandmarks[0].y; 
        const xDiff = Math.abs(rightLandmarks[0].x - leftLandmarks[0].x) / scale;
        if (yDiff < -0.1 && xDiff < 0.4) detected = "G"; 
      }

      // H: Right flat hand sweeping left flat hand
      else if (lOpen && rOpen) {
        const palmDist = this.getDistance(leftLandmarks[0], rightLandmarks[0]) / scale;
        if (palmDist < 0.5) detected = "H";
      }

      // K: Right index bent touching left index straight
      else if (lIndexUp && !lMiddleUp && rIndexUp && !rMiddleUp) {
        // Right index bent implies tip is lower than pip
        const rIndexBent = rightLandmarks[8].y > rightLandmarks[6].y;
        if (rIndexBent && this.getDistance(rightLandmarks[6], leftLandmarks[6]) / scale < 0.3) {
          detected = "K";
        }
      }

      // L: Right L shape on left palm
      else if (lOpen && rIndexUp && !rMiddleUp && !rRingUp && !rPinkyUp) {
        const rThumbExt = this.getDistance(rightLandmarks[4], rightLandmarks[5]) / scale > 0.4;
        const touchPalm = this.getDistance(rightLandmarks[8], leftLandmarks[9]) / scale < 0.3;
        if (rThumbExt && touchPalm) detected = "L";
      }

      // M: Right three fingers (index, middle, ring) touching left palm
      else if (lOpen && rIndexUp && rMiddleUp && rRingUp && !rPinkyUp) {
        const touchPalm = this.getDistance(rightLandmarks[12], leftLandmarks[9]) / scale < 0.3;
        if (touchPalm) detected = "M";
      }

      // N: Right two fingers (index, middle) touching left palm
      else if (lOpen && rIndexUp && rMiddleUp && !rRingUp && !rPinkyUp) {
        const touchPalm = this.getDistance(rightLandmarks[12], leftLandmarks[9]) / scale < 0.3;
        if (touchPalm) detected = "N";
      }

      // P: Right circle (index+thumb) touching left index tip
      else if (lIndexUp && !lMiddleUp && rThumbIndexClose && !rMiddleUp) {
        const touchTip = this.getDistance(rightLandmarks[8], leftLandmarks[8]) / scale < 0.3;
        if (touchTip) detected = "P";
      }

      // S: Pinkies hooked
      else if (lFist && rFist && lPinkyUp && rPinkyUp) {
        const hook = this.getDistance(leftLandmarks[20], rightLandmarks[20]) / scale < 0.2;
        if (hook) detected = "S";
      }

      // T: Right index touching lower left palm
      else if (lOpen && rIndexUp && !rMiddleUp && !rRingUp) {
        const touchWrist = this.getDistance(rightLandmarks[8], leftLandmarks[0]) / scale < 0.3;
        if (touchWrist) detected = "T";
      }

      // V: Right V shape on left palm
      else if (lOpen && rIndexUp && rMiddleUp && !rRingUp) {
        const vSeparation = this.getDistance(rightLandmarks[8], rightLandmarks[12]) / scale > 0.2;
        const touchPalm = this.getDistance(rightLandmarks[10], leftLandmarks[9]) / scale < 0.3;
        if (vSeparation && touchPalm) detected = "V";
      }

      // W: Fingers interlocked
      else if (lOpen && rOpen) {
        const indexToMiddle = this.getDistance(leftLandmarks[8], rightLandmarks[12]) / scale < 0.2;
        const middleToIndex = this.getDistance(leftLandmarks[12], rightLandmarks[8]) / scale < 0.2;
        if (indexToMiddle && middleToIndex) detected = "W";
      }

      // X: Index fingers crossed
      else if (lIndexUp && !lMiddleUp && rIndexUp && !rMiddleUp) {
        const cross = this.getDistance(leftLandmarks[6], rightLandmarks[6]) / scale < 0.2;
        if (cross) detected = "X";
      }
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
