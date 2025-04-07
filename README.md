# rl_avellaneda_stoikov

Репозиторий с моделью avellaneda stoikov дополненной ppo для подбора оптимальных параметров $\gamma$ и k. Решение исключительно как пайплайн, улучшению подлежит абсолютно все. 
Для запуска достаточно заменить .env.example и docker-compose.yaml.example и при желании убрать volumes 
(возможно в ближайшее время и образ залью в докерхаб).






### TODO
1) Поменять фичи на вход
2) Награда для краткосрочных целей sharp не лучшая идея
3) Параметры модели накинуть подбор
4) Сделать модели для дисперсии и волатильности и подавать предикшн на вход ppo
5) Сделать наглядное отображение результатов
6) Сделать версию для торгов

P.S. Я считаю не очень удачной идеей применять RL в данном случае, т.к. RL уместен в двух случаях (1) есть среда для обучения (2) нет таргета или его тяжело построить.

1) Например в играх есть воспроизводимая среда, поэтому RL идеален для игр, также среду воссоздают с основными физическими законами для обучения роботов или автопилотов.

2) В случае с rlhf у llm-ок этот подход применяется ввиду отсутсвия таргета, т.к. llm обучается предсказывать следующее слово, писать новый качественный текст после претрейна долго, сложно и дорого, чтобы обогатить этот подход используют ранжирование ответов в качестве фидбека от пользователей.

В случае же с маркетмейкингом все вертится вокруг ожидаемой волатильности (в нашей модели это $\gamma$ как уровень риска или напрямую $\sigma^2$) и ликвидности (в нашей модели для этого есть k), нужно их корректно оценить. (здесь и далее предполагаю название параметров как из оригинальной статьи или формулы ниже)

$$
r(s, q, t) = s - q \gamma \sigma^2 (T - t)
$$

$$
\delta^a + \delta^b = \frac{2}{\gamma} \ln \left(1 + \frac{\gamma}{k}\right)
$$


И в данной задаче оба пункта и про среду и про таргет мимо:

(1) Для этого не очень удобная среда, т.к. на исторических данных мы не можем оценить свое влияние на рынок, а в реалтайме не так много параллельных торгов можно запустить, они будут влиять друг на друга, да и деньги терять не охото запуская множество агентов параллельно ведущих торги.

(2) Таргет без проблем можно посчитать просто взяв значения дисперсии и объема торгов (или их отнормированные аналоги) за будущий(относительно фичей) период времени и исходя из их значений поменять параметры $\gamma$ и k, насколько нужно менять параметры зная дисперсию и волатильность можно откалибровать на исторических данных (просто разбить по бакетам волатильности и ликвидности и в цикле перебором подобрать оптимальные параметры, после чего ml модели будут пресказывать ликвидность и волатильность, а дальше выставлять оптимальные параметры).

Если хочется добавить некоторой динамики, чтобы модель обучалась и не стояла на месте можно поставить её на автомониторинг: оценивать дрейф распределения фичей через какой-нибудь тест вроде хи-квадрат и средний выдаваемый результат и когда происходит статзначимое изменение запускать переобучение модели на новой выборке. Такой подход будет существенно более робастным, надежным и прибыльным. Имхо модели в RL имеют очень большое количество недостатков в сравнении с обычным мл и должны применяться только при отсутствии выбора.

