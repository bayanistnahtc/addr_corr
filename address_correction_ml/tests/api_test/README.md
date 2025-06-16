### API тесты (Сервис Корректировки Адресов)

На основе `configuration.json.j2` подготовить конфиг `configuration.json` с адресами сервисов (Triton, Address Correction Service)

#### Запуск и настройка окружения
Для управления зависимостями используем виртуальную среду `pipenv`
```shell
pip install pipenv
pipenv shell    # запуск виртуальной среды
pipenv install  # установка зависимостей из Pipfile
```

#### Запуск тестов

```shell
pipenv run pytest
```

#### Запуск тестов с Allure отчетом, [wiki](https://wiki.maxim-team.ru/pages/viewpage.action?pageId=243499566)

```shell
pipenv run pytest --alluredir=allure-results
allure serve
```
