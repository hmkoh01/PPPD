# Mobile Team Demo

This project can be shared through one HTTPS Cloudflare Quick Tunnel URL.
Only the Next.js frontend is exposed. The FastAPI backend stays on the Docker
internal network and is reached through Next.js rewrites.

## Structure

- Frontend: `frontend`
- Backend: `backend`
- Next.js config: `frontend/next.config.ts`
- Frontend API client: `frontend/lib/api.ts`
- Frontend image URL helper: `frontend/lib/image.ts`
- FastAPI entrypoint: `backend/app/main.py`, app object `app`
- Backend dependencies: `backend/requirements.txt`

## Run With Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f
```

Frontend:

```txt
http://localhost:3000
```

Proxy health check:

```txt
http://localhost:3000/api/health
```

Expected response:

```json
{"status":"ok"}
```

## Cloudflare Quick Tunnel

Run this on the host machine after Docker Compose is up:

```bash
cloudflared tunnel --url http://localhost:3000
```

Share the generated URL with the team:

```txt
https://random-words.trycloudflare.com
```

The UI and API both use the same public origin. Browser requests go to
`https://random-words.trycloudflare.com/api/...`, Next.js proxies those requests
to the backend service at `http://backend:8000`.

## Mobile Test Checklist

1. Open the Cloudflare HTTPS link.
2. Confirm the page renders.
3. Open `/api/health` on the same Cloudflare origin.
4. Allow camera permission.
5. Capture a photo.
6. Upload the photo and confirm API processing.
7. Confirm the result screen.

## Troubleshooting

### Page Loads But API Fails

Search for hardcoded backend URLs:

```bash
grep -R "localhost:8000" frontend
grep -R "127.0.0.1:8000" frontend
```

Browser-side frontend code should call same-origin paths such as `/api/...`.

### `/api/health` Is 404

- Check `frontend/next.config.ts`.
- Rebuild the Docker image:

```bash
docker compose up -d --build
```

### `/api/health` Is 502 Or Connection Fails

- Check that the Compose service is named `backend`.
- Check that the frontend has `BACKEND_URL=http://backend:8000`.
- Check backend logs:

```bash
docker compose logs -f backend
```

### Mobile Camera Does Not Work

- Use the Cloudflare HTTPS URL, not plain HTTP.
- Check browser camera permission.
- Test separately on iOS Safari and Android Chrome.
