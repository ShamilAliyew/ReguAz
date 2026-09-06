# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# The browser talks to Nginx on the same origin. Nginx proxies /api requests
# to the private backend service, so no host-specific URL is baked in.
ARG VITE_API_URL=/api
ARG VITE_USE_MOCK_API=false
ENV VITE_API_URL=${VITE_API_URL} \
    VITE_USE_MOCK_API=${VITE_USE_MOCK_API}

RUN npm run build


FROM nginx:1.27-alpine AS runtime

ENV NGINX_ENVSUBST_FILTER="^(PORT|API_UPSTREAM)$"

COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD wget -qO- "http://127.0.0.1:${PORT:-80}/" >/dev/null || exit 1
