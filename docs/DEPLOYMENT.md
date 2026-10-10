# 🚀 Руководство по деплою в продакшн

Это руководство поможет вам развернуть систему в production окружении.

## 📋 Варианты деплоя

### 1. Docker Compose (для разработки и небольших проектов)

**Преимущества:**
- ✅ Простота настройки
- ✅ Быстрый запуск
- ✅ Подходит для тестирования

**Недостатки:**
- ❌ Не масштабируется
- ❌ Один сервер

**Инструкция:**
```bash
# 1. Клонировать репозиторий
git clone https://github.com/AntoGA/Vk.git
cd Vk

# 2. Настроить .env
cp .env.example .env
nano .env  # Добавить токены и настройки

# 3. Собрать образы
docker-compose build

# 4. Запустить сервисы
docker-compose up -d

# 5. Проверить статус
docker-compose ps
docker-compose logs -f api

# 6. Открыть в браузере
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

**Масштабирование:**
```bash
# Увеличить количество API инстансов
docker-compose up -d --scale api=3
```

---

### 2. Kubernetes (для production)

**Преимущества:**
- ✅ Автоматическое масштабирование
- ✅ Высокая доступность
- ✅ Self-healing
- ✅ Rolling updates

**Недостатки:**
- ❌ Сложность настройки
- ❌ Требует Kubernetes кластер

**Предварительные требования:**
- Kubernetes кластер (v1.25+)
- kubectl настроен
- Helm (опционально)
- Container registry (Docker Hub, GitHub Container Registry)

**Инструкция:**

#### Шаг 1: Собрать и запушить Docker образ
```bash
# Собрать образ
docker build -t your-registry/vk-interest-predictor:latest .

# Запушить в registry
docker push your-registry/vk-interest-predictor:latest
```

#### Шаг 2: Настроить Kubernetes манифесты
```bash
# Отредактировать секреты
nano k8s/secret.yaml
# Добавить VK_API_TOKEN, DATABASE_URL, REDIS_PASSWORD

# Отредактировать ConfigMap (если нужно)
nano k8s/configmap.yaml
```

#### Шаг 3: Применить манифесты
```bash
# Создать namespace
kubectl apply -f k8s/namespace.yaml

# Создать ConfigMap и Secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Запустить Redis
kubectl apply -f k8s/redis-deployment.yaml

# Запустить API
kubectl apply -f k8s/api-deployment.yaml

# Запустить Dashboard
kubectl apply -f k8s/dashboard-deployment.yaml

# Настроить Ingress
kubectl apply -f k8s/ingress.yaml
```

#### Шаг 4: Проверить деплой
```bash
# Проверить поды
kubectl get pods -n vk-interest-predictor

# Проверить сервисы
kubectl get svc -n vk-interest-predictor

# Проверить логи
kubectl logs -f deployment/api -n vk-interest-predictor

# Получить внешний IP
kubectl get ingress -n vk-interest-predictor
```

**Масштабирование:**
```bash
# Увеличить количество реплик API
kubectl scale deployment api --replicas=5 -n vk-interest-predictor

# Автоматическое масштабирование (HPA)
kubectl autoscale deployment api --cpu-percent=50 --min=3 --max=10 -n vk-interest-predictor
```

---

### 3. Yandex Cloud

**Преимущества:**
- ✅ Российский провайдер
- ✅ Интеграция с VK
- ✅ Managed Kubernetes

**Инструкция:**

#### Шаг 1: Создать инфраструктуру
```bash
# Установить Yandex Cloud CLI
curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash

# Инициализировать
yc init

# Создать Kubernetes кластер
yc managed-kubernetes cluster create \
  --name vk-cluster \
  --zone ru-central1-a \
  --network-id your-network-id \
  --subnet-id your-subnet-id

# Получить kubeconfig
yc managed-kubernetes cluster get-credentials --name vk-cluster --external
```

#### Шаг 2: Задеплоить приложение
```bash
# Использовать скрипт деплоя
python scripts/deploy.py --method yandex-cloud --project-id your-project-id
```

---

### 4. AWS / Google Cloud / Azure

**Инструкция аналогична Kubernetes деплою:**

1. Создать Kubernetes кластер (EKS / GKE / AKS)
2. Настроить kubectl
3. Применить манифесты из `k8s/`

**AWS EKS:**
```bash
# Создать кластер
eksctl create cluster --name vk-cluster --region us-east-1

# Задеплоить
kubectl apply -f k8s/
```

**Google GKE:**
```bash
# Создать кластер
gcloud container clusters create vk-cluster --zone us-central1-a

# Задеплоить
kubectl apply -f k8s/
```

---

## 🔧 CI/CD Pipeline (GitHub Actions)

**Автоматический деплой при push в main:**

Файл `.github/workflows/ci-cd.yml` уже настроен и выполняет:

1. **Test** — запуск тестов на Python 3.11 и 3.12
2. **Lint** — проверка кода (black, flake8, mypy)
3. **Build** — сборка Docker образа
4. **Push** — публикация в Container Registry
5. **Deploy** — деплой в Kubernetes (если настроено)

**Настройка секретов в GitHub:**
1. Перейти в Settings → Secrets and variables → Actions
2. Добавить:
   - `DOCKER_USERNAME` — логин Docker Hub
   - `DOCKER_PASSWORD` — пароль Docker Hub
   - `KUBE_CONFIG` — base64-encoded kubeconfig
   - `VK_API_TOKEN` — токен VK API

**Ручной запуск:**
```bash
# Trigger workflow вручную
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/AntoGA/Vk/actions/workflows/ci-cd.yml/dispatches \
  -d '{"ref":"main"}'
```

---

## 📊 Мониторинг и логирование

### Prometheus + Grafana

**Установка:**
```bash
# Добавить Helm репозиторий
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Установить Prometheus Stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Получить пароль Grafana
kubectl get secret --namespace monitoring monitoring-grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Открыть Grafana
kubectl port-forward svc/monitoring-grafana 3000:80 --namespace monitoring
# http://localhost:3000 (admin / пароль выше)
```

**Метрики для мониторинга:**
- Latency (p50, p95, p99)
- Throughput (requests/sec)
- Error rate
- CPU/Memory usage
- GPU utilization (если используется)
- Model accuracy

### ELK Stack (Elasticsearch + Logstash + Kibana)

**Установка:**
```bash
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch --namespace logging --create-namespace
helm install kibana elastic/kibana --namespace logging
```

---

## 🔐 Безопасность

### SSL/TLS сертификаты

**Let's Encrypt (автоматически):**
```bash
# Установить cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Создать ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Network Policies

**Ограничить доступ между подами:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: vk-interest-predictor
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: dashboard
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
```

---

## 📈 Performance Tuning

### Оптимизация модели

```bash
# Конвертировать в ONNX для быстрого inference
python -c "
import torch
from ml.models import InterestPredictor

model = InterestPredictor.load_from_checkpoint('models/best_model.pt')
model.eval()

example_input = torch.randn(1, 100, 256)
torch.onnx.export(
    model,
    example_input,
    'models/best_model.onnx',
    opset_version=11
)
"

# Использовать ONNX Runtime
pip install onnxruntime
```

### Кэширование

**Redis настройки:**
```yaml
# configs/inference.yaml
caching:
  enabled: true
  ttl: 300  # 5 минут
  max_size: 10000
```

### Load Balancing

**Nginx Ingress Controller:**
```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx
```

---

## 🔄 Rolling Updates

**Обновление без downtime:**
```bash
# Обновить образ
docker build -t your-registry/vk-interest-predictor:v2 .
docker push your-registry/vk-interest-predictor:v2

# Обновить deployment
kubectl set image deployment/api api=your-registry/vk-interest-predictor:v2 -n vk-interest-predictor

# Проверить статус
kubectl rollout status deployment/api -n vk-interest-predictor

# Откатить (если проблема)
kubectl rollout undo deployment/api -n vk-interest-predictor
```

---

## 🆘 Troubleshooting

### Проблема: Pod в состоянии CrashLoopBackOff
```bash
# Посмотреть логи
kubectl logs deployment/api -n vk-interest-predictor

# Посмотреть события
kubectl describe pod -l app=api -n vk-interest-predictor
```

### Проблема: Не подключается к Redis
```bash
# Проверить Redis pod
kubectl get pods -l app=redis -n vk-interest-predictor

# Проверить сервис
kubectl get svc redis-service -n vk-interest-predictor

# Проверить connectivity
kubectl exec -it deployment/api -- redis-cli -h redis-service ping
```

### Проблема: Высокая latency
```bash
# Проверить ресурсы
kubectl top pods -n vk-interest-predictor

# Увеличить реплики
kubectl scale deployment api --replicas=5 -n vk-interest-predictor

# Проверить логи
kubectl logs deployment/api -n vk-interest-predictor | grep "inference_time"
```

---

## 📚 Дополнительные ресурсы

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Yandex Cloud Documentation](https://cloud.yandex.ru/docs)
