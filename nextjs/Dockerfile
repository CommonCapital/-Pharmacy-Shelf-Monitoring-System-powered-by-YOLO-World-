FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Build step skipped for dev mode, but included for completeness
# RUN npm run build

CMD ["npm", "run", "dev"]
