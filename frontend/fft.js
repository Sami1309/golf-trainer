// In-place radix-2 Cooley-Tukey FFT. N must be a power of 2.
// real[], imag[] are Float32Array of length N. imag is zeroed for real input.
export function fft(real, imag) {
  const N = real.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = real[i]; real[i] = real[j]; real[j] = t;
      t = imag[i]; imag[i] = imag[j]; imag[j] = t;
    }
  }
  // Butterflies
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const ang = -2 * Math.PI / len;
    const wCos = Math.cos(ang), wSin = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let cCos = 1, cSin = 0;
      for (let k = 0; k < half; k++) {
        const aRe = real[i + k], aIm = imag[i + k];
        const bRe = real[i + k + half], bIm = imag[i + k + half];
        const tRe = cCos * bRe - cSin * bIm;
        const tIm = cCos * bIm + cSin * bRe;
        real[i + k] = aRe + tRe;
        imag[i + k] = aIm + tIm;
        real[i + k + half] = aRe - tRe;
        imag[i + k + half] = aIm - tIm;
        const nCos = cCos * wCos - cSin * wSin;
        const nSin = cCos * wSin + cSin * wCos;
        cCos = nCos; cSin = nSin;
      }
    }
  }
}

// Compute magnitude spectrum (length N/2) of a real signal of length N.
// `frame` is a Float32Array of length N (windowed as needed by caller).
export function magSpectrum(frame) {
  const N = frame.length;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re.set(frame);
  fft(re, im);
  const mag = new Float32Array(N >> 1);
  for (let k = 0; k < mag.length; k++) {
    mag[k] = Math.hypot(re[k], im[k]);
  }
  return mag;
}

// Hann window of length N.
export function hann(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
  return w;
}
