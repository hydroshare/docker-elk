# docker-compose -f docker-compose.yml -f docker-compose-prod.yml run --no-deps  certbot  renew
docker compose -f docker-compose-prod.yml up certbot
docker compose exec nginx nginx -s reload