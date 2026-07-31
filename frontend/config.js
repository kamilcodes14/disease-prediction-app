// Point this at a separate backend URL only if you deploy the backend
// elsewhere (e.g. Railway/Render). For this project the backend lives
// on the same Vercel domain, so leaving this empty is correct — predict
// calls just use a relative path like /api/predict/image.
window.API_BASE = window.API_BASE || "";