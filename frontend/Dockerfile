# Lightweight nginx to serve the static TRACE frontend
FROM nginx:1.27-alpine

# Remove the default nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy our static frontend file as the site root
COPY index.html /usr/share/nginx/html/index.html

# Copy custom nginx config (handles SPA routing + correct MIME types)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

# nginx reads config from /etc/nginx/conf.d/default.conf automatically
CMD ["nginx", "-g", "daemon off;"]
