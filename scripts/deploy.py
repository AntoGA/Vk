#!/usr/bin/env python3
"""
Скрипт для деплоя в production.

Использование:
    # Деплой с Docker Compose
    python scripts/deploy.py --method docker-compose

    # Деплой с Kubernetes
    python scripts/deploy.py --method kubernetes

    # Деплой на Yandex Cloud
    python scripts/deploy.py --method yandex-cloud --project-id my-project
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, cwd: Path = None) -> bool:
    """Выполняет команду и возвращает True при успехе."""
    logger.info(f"Выполнение: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка выполнения команды: {e}")
        if e.stderr:
            logger.error(e.stderr)
        return False


def deploy_docker_compose(project_root: Path):
    """Деплой с Docker Compose."""
    logger.info("🐳 Запуск деплоя с Docker Compose...")
    
    # Проверяем наличие docker-compose.yml
    compose_file = project_root / "docker-compose.yml"
    if not compose_file.exists():
        logger.error("Файл docker-compose.yml не найден")
        return False
    
    # Останавливаем старые контейнеры
    logger.info("Остановка старых контейнеров...")
    run_command(["docker-compose", "down"], cwd=project_root)
    
    # Собираем образы
    logger.info("Сборка Docker образов...")
    if not run_command(["docker-compose", "build"], cwd=project_root):
        return False
    
    # Запускаем контейнеры
    logger.info("Запуск контейнеров...")
    if not run_command(["docker-compose", "up", "-d"], cwd=project_root):
        return False
    
    logger.info("✅ Docker Compose деплой завершен успешно!")
    logger.info("Сервисы доступны:")
    logger.info("  - API: http://localhost:8000")
    logger.info("  - Dashboard: http://localhost:8501")
    logger.info("  - Prometheus: http://localhost:9090")
    logger.info("  - Grafana: http://localhost:3000")
    
    return True


def deploy_kubernetes(project_root: Path):
    """Деплой с Kubernetes."""
    logger.info("☸️ Запуск деплоя с Kubernetes...")
    
    k8s_dir = project_root / "k8s"
    if not k8s_dir.exists():
        logger.error("Директория k8s/ не найдена")
        return False
    
    # Применяем манифесты
    manifests = [
        "namespace.yaml",
        "configmap.yaml",
        "secret.yaml",
        "redis-deployment.yaml",
        "api-deployment.yaml",
        "dashboard-deployment.yaml",
        "ingress.yaml"
    ]
    
    for manifest in manifests:
        manifest_path = k8s_dir / manifest
        if manifest_path.exists():
            logger.info(f"Применение {manifest}...")
            if not run_command(["kubectl", "apply", "-f", str(manifest_path)]):
                return False
    
    logger.info("✅ Kubernetes деплой завершен успешно!")
    logger.info("Проверьте статус: kubectl get pods -n vk-interest-predictor")
    
    return True


def deploy_yandex_cloud(project_root: Path, project_id: str):
    """Деплой на Yandex Cloud."""
    logger.info("☁️ Запуск деплоя на Yandex Cloud...")
    
    # Проверяем наличие yc CLI
    if not run_command(["yc", "version"]):
        logger.error("Yandex Cloud CLI (yc) не установлен")
        logger.info("Установите: curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash")
        return False
    
    # Создаем Container Registry (если не существует)
    logger.info("Создание Container Registry...")
    run_command([
        "yc", "container", "registry", "create",
        "--name", "vk-interest-predictor",
        "--folder-id", project_id
    ])
    
    # Собираем и пушим Docker образ
    logger.info("Сборка и публикация Docker образа...")
    registry_id = subprocess.check_output([
        "yc", "container", "registry", "get", "vk-interest-predictor",
        "--folder-id", project_id, "--format", "json"
    ]).decode()
    
    import json
    registry_data = json.loads(registry_id)
    registry_name = registry_data["name"]
    
    image_name = f"cr.yandex/{registry_name}/vk-api:latest"
    
    if not run_command(["docker", "build", "-t", image_name, "."], cwd=project_root):
        return False
    
    if not run_command(["docker", "push", image_name]):
        return False
    
    # Деплоим в Serverless Containers
    logger.info("Деплой в Yandex Serverless Containers...")
    if not run_command([
        "yc", "serverless", "container", "create",
        "--name", "vk-interest-predictor",
        "--folder-id", project_id,
        "--image", image_name,
        "--cores", "2",
        "--memory", "2g",
        "--concurrency", "10",
        "--execution-timeout", "60s"
    ]):
        return False
    
    logger.info("✅ Yandex Cloud деплой завершен успешно!")
    logger.info("Проверьте статус: yc serverless container list")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Деплой VK Interest Predictor")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["docker-compose", "kubernetes", "yandex-cloud"],
        help="Метод деплоя"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="ID проекта в Yandex Cloud (обязательно для yandex-cloud)"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    if args.method == "docker-compose":
        success = deploy_docker_compose(project_root)
    elif args.method == "kubernetes":
        success = deploy_kubernetes(project_root)
    elif args.method == "yandex-cloud":
        if not args.project_id:
            logger.error("Для Yandex Cloud необходимо указать --project-id")
            sys.exit(1)
        success = deploy_yandex_cloud(project_root, args.project_id)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
