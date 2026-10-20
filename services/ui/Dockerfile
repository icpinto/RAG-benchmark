FROM node:22-alpine AS build
WORKDIR /app
COPY services/ui/package.json services/ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY services/ui/ ./
RUN npm run build

FROM nginx:stable-alpine
COPY services/ui/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
ENV API_BASE_URL=http://api:8000
EXPOSE 80
