---
title: "Semantic IDs для музыкальной рекомендации"
author: "[Имя Фамилия / команда]"
date: "[дата]"
---

# Мини-абстракт

В проекте исследуется, как способы построения semantic IDs влияют на качество музыкальной рекомендации. Моя часть состоит из двух связанных экспериментов.

Первый эксперимент проверяет, помогает ли artist/album supervision строить более полезные и интерпретируемые variable-length semantic IDs на Yambda. Мы сравниваем оригинальный VarLen dVAE с вариантами, где artist/album metadata добавляется через auxiliary loss или через supervision первых токенов SID.

Второй эксперимент проверяет, кому semantic IDs помогают сильнее. Гипотеза состоит в том, что VarLen SID должны давать больший прирост пользователям с длинной историей, потому что у модели есть больше контекста для понимания вкуса. Этот эксперимент сравнивает fixed-length dVAE и VarLen dVAE на Yambda и Amazon.

## Почему не full reproduction TIGER

Изначальная постановка была: воспроизвести статью [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) и провести свой RQ. В этой статье предлагается generative retrieval paradigm: item получает Semantic ID как последовательность codewords, а Transformer по истории пользователя autoregressively предсказывает Semantic ID следующего item.

Мы сохраняем эту базовую идею — recommendation через генерацию Semantic ID, — но не делаем full reproduction TIGER по нескольким причинам.

Во-первых, наш RQ уже не про проверку всего TIGER end-to-end, а про более узкий вопрос: **может ли artist/album supervision улучшить variable-length semantic IDs для музыкальной рекомендации?** Для этого нужен пайплайн, где variable-length SID и SID diagnostics являются центральной частью реализации.

Во-вторых, `KhrylchenkoKirill/varlen_semantic_ids` лучше соответствует этому RQ как экспериментальная база: в нем уже есть VarLen dVAE, fixed/varlen comparison, seqrec evaluation, Yambda preprocessing и config-driven запуск. Поэтому его проще расширить аккуратными loss-вариантами и structural metrics, не переписывая весь generative retrieval stack.

В-третьих, full reproduction paper-scale TIGER плохо совпадает с compute budget проекта. Для курса важнее получить небольшой, воспроизводимый и честно интерпретируемый эксперимент, чем частично воспроизвести большой pipeline без возможности полноценно проверить все baselines.

Итого: переход к другой репе — это не смена темы, а сужение scope. TIGER остается conceptual baseline, а `varlen_semantic_ids` используется как практический scaffold для RQ про VarLen SID и artist/album supervision.

# Общий сетап и термины

## Что такое semantic ID

Semantic ID, или SID, — это дискретный код айтема. Вместо того чтобы использовать обычный item id, мы представляем айтем последовательностью токенов:

```text
item -> [sid_1, sid_2, ..., sid_L]
```

Смысл SID в том, что похожие айтемы могут получать похожие коды или общие префиксы. Это удобно для генеративной рекомендации: seqrec-модель предсказывает не item id напрямую, а последовательность SID-токенов.

В fixed-length варианте длина SID постоянна. В variable-length варианте модель сама выбирает длину кода, поэтому простые или популярные айтемы могут получать более короткие ID, а сложные — более длинные.

## Данные

В экспериментах используются:

- **Yambda** для artist/album supervision и history-length анализа;
- **Amazon** для дополнительной проверки history-length гипотезы.

Для Yambda доступны:

- user-item interactions;
- item embeddings;
- artist-item mapping;
- album-item mapping.

Для Amazon используются:

- user-item interactions;
- item embeddings;
- temporal train/test split.

Подготовка данных следует temporal protocol исходного пайплайна: train берется из более раннего периода, test — из более позднего. Для обучения SID используются core items после фильтрации по минимальной частоте.

## SID holdout fraction

`SID holdout fraction = 0.1` означает, что 10% core items откладываются как holdout при диагностике качества SID. Это не test split для sequential recommendation. Этот holdout нужен, чтобы проверять, насколько хорошо SID-модель обобщается на айтемы, которые не использовались как train items при SID-обучении.

В режиме `train_holdout`, который используется для downstream seqrec, SID-модель затем обучается на train+holdout items, чтобы получить коды для полного core-каталога.

## Seqrec evaluation

После построения SID обучается облегченная sequential recommendation модель. Она получает историю пользователя как последовательность SID-токенов и должна предсказать следующий понравившийся айтем.

Seqrec-конфигурация намеренно меньше paper-scale setup:

| Параметр | Значение |
|---|---:|
| History budget | 128 tokens |
| Transformer depth | 4 |
| Beam size | 50 |
| Cutoffs | @10, @50 |

Поэтому абсолютные значения Recall/NDCG не являются reproduction статьи. Основная цель — относительное сравнение методов в одинаковом lightweight setup.

`@100` в текущем запуске намеренно не репортится. В seqrec-конфигах стоит `beam_size: 50`, а `append_unique_token: true` добавляет item-specific токен к SID и деколлизирует выдачу для downstream evaluation. В результате beam search возвращает ровно 50 SID-кандидатов на пользователя (`unique_sids_per_user = 50.0` в `seqrec_summary.json`), поэтому `Recall@100`, `NDCG@100` и `Coverage@100` совпадают с `@50` и не несут дополнительной информации.

## Метрики recommendation quality

| Метрика | Что означает |
|---|---|
| Recall@K | Доля target items, попавших в top-K рекомендаций |
| NDCG@K | Учитывает не только попадание target item, но и позицию в ranking |
| Coverage@K | Какая доля каталога появляется среди top-K рекомендаций |
| Head/tail metrics | Отдельное качество на популярных и менее популярных айтемах |

## Structural SID metrics

| Метрика | Что означает |
|---|---|
| Mean SID Length | Средняя длина SID |
| Unique SIDs | Сколько уникальных кодов получилось |
| Collision | Ситуация, когда несколько айтемов получили один и тот же SID |
| Excess Collisions | Сколько айтемов “лишние” внутри collision buckets |
| Max Bucket Size | Максимальное число айтемов с одинаковым SID |
| Prefix purity | Насколько айтемы с одинаковым SID-префиксом принадлежат одному artist/album |

**Prefix purity** считается так: для каждого префикса SID берутся все айтемы с этим префиксом, внутри bucket-а находится самый частый artist или album, затем считается доля айтемов, принадлежащих этому большинству. Чем выше purity, тем лучше SID-префикс согласован с metadata.

Например, `Artist Purity@1` показывает, насколько первый SID-токен соответствует группировке по артисту. `Album Purity@2` показывает, насколько первые два SID-токена соответствуют группировке по альбому.

# Эксперимент 1: artist/album supervision

## Research question

**Помогает ли artist/album supervision строить более качественные variable-length semantic IDs для музыкальной рекомендации?**

Под качеством понимается:

- downstream recommendation quality;
- структурная осмысленность SID-префиксов;
- умеренное число коллизий.

## Методы

### dVAE fixed-length (`fixed`)

Это базовый dVAE, который всегда строит SID фиксированной длины `L = 5`.

Модель состоит из encoder и decoder:

- encoder переводит item embedding в последовательность дискретных токенов;
- decoder пытается восстановить item embedding по этим токенам.

Все айтемы получают SID одинаковой длины. Artist/album metadata не используется. Этот метод нужен как простой baseline: он показывает, что дает обычный дискретный код без variable-length механизма.

### dVAE variable-length (`original`)

Это основной baseline из статьи. В отличие от fixed-length dVAE, модель может выбирать длину SID от 1 до 5.

В loss есть reconstruction term и регуляризация длины. Модель получает стимул делать SID короче, если короткого кода достаточно для восстановления embedding. Artist/album metadata не используется.

Этот метод отвечает на вопрос: что дает сама variable-length структура без дополнительной metadata.

### dVAE variable-length + auxiliary artist/album loss (`aux`)

Это VarLen dVAE, к которому добавлены две вспомогательные classification задачи:

- предсказать artist cluster;
- предсказать album cluster.

Предсказание делается по агрегированному представлению всего SID. То есть metadata supervision действует мягко на весь код, но не требует, чтобы конкретный токен отвечал за artist или album.

Интуиция: artist/album loss может работать как регуляризация. Он подталкивает SID хранить информацию, полезную для музыкальной структуры, но оставляет модели свободу самой решить, где именно в коде эту информацию разместить.

### dVAE variable-length + prefix artist/album loss (`prefix`)

Это более жесткий metadata-aware вариант.

Здесь supervision привязан к началу SID:

- первый токен SID должен помогать предсказывать artist;
- первые два токена SID должны помогать предсказывать album.

Интуиция: если SID используется как последовательность токенов, то ранние токены особенно важны. Они задают крупную группу кандидатов в beam search. Поэтому полезно, чтобы префикс явно отражал artist/album structure.

Потенциальный риск: слишком сильное давление на префикс может ухудшить reconstruction или downstream recommendation, если artist/album структура не совпадает с тем, что нужно для предсказания лайков.

### residual KMeans SID (`rkmeans`)

Это не dVAE, а residual quantization baseline.

Метод строит SID последовательно:

1. первый k-means кодирует embedding;
2. затем считается residual;
3. следующий k-means кодирует residual;
4. процесс повторяется несколько уровней.

Artist/album metadata не используется. Этот baseline нужен, чтобы отделить эффект metadata-aware структуры от обычной residual quantization.

### hierarchical artist/album R-KMeans (`hier`)

Это hard metadata-anchored baseline. SID строится вручную как:

```text
[artist_cluster, album_cluster, residual_1, residual_2, residual_3]
```

Сначала артисты кластеризуются по средним embedding-ам. Затем внутри artist cluster альбомы кластеризуются локально. После этого оставшийся residual embedding кодируется обычным residual k-means.

Этот метод проверяет крайний вариант: что будет, если artist/album structure не учить мягко, а зашить прямо в первые SID-токены.

## Эксперимент 1.1: downstream recommendation quality

**Цель.** Проверить, улучшают ли artist/album-aware SID качество sequential recommendation.

| Метод | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Coverage@50 |
|---|---:|---:|---:|---:|---:|
| dVAE fixed-length | 0.0319 | 0.0830 | 0.0161 | 0.0272 | 0.0704 |
| dVAE varlen | 0.0397 | 0.1033 | 0.0203 | 0.0342 | 0.1109 |
| dVAE varlen + aux loss | **0.0457** | **0.1171** | **0.0235** | **0.0390** | **0.1417** |
| dVAE varlen + prefix loss | 0.0444 | 0.1104 | 0.0223 | 0.0366 | 0.1261 |
| Residual KMeans | 0.0299 | 0.0821 | 0.0154 | 0.0267 | 0.0615 |
| Hier. artist/album R-KMeans | 0.0427 | 0.1050 | 0.0225 | 0.0360 | 0.1340 |

Head/tail разрез:

| Метод | Head Recall@50 | Head NDCG@50 | Tail Recall@50 | Tail NDCG@50 | Long-tail share |
|---|---:|---:|---:|---:|---:|
| dVAE fixed-length | 0.1115 | 0.0369 | 0.0071 | 0.0015 | 0.0289 |
| dVAE varlen | 0.1369 | 0.0459 | 0.0139 | 0.0030 | 0.0451 |
| dVAE varlen + aux loss | **0.1526** | **0.0516** | 0.0225 | 0.0054 | **0.0598** |
| dVAE varlen + prefix loss | 0.1467 | 0.0491 | 0.0139 | 0.0033 | 0.0559 |
| Residual KMeans | 0.1113 | 0.0364 | 0.0044 | 0.0010 | 0.0207 |
| Hier. artist/album R-KMeans | 0.1355 | 0.0468 | **0.0241** | **0.0070** | 0.0469 |

**Выводы.**

В этом lightweight setup лучший overall результат дает dVAE varlen + aux loss: относительно dVAE varlen он прибавляет `+0.0059` Recall@10, `+0.0138` Recall@50, `+0.0032` NDCG@10 и `+0.0048` NDCG@50. Это поддерживает гипотезу, что artist/album supervision может быть полезной регуляризацией для VarLen dVAE.

dVAE varlen + prefix loss тоже лучше dVAE varlen, но слабее aux loss: `+0.0047` Recall@10 и `+0.0071` Recall@50. Жесткая привязка metadata к ранним токенам повышает artist-level интерпретируемость, но не дает максимального downstream качества.

Hierarchical artist/album R-KMeans улучшает dVAE varlen совсем умеренно по Recall/NDCG, зато дает лучший tail Recall@50 и tail NDCG@50. Это похоже на эффект более явной metadata-структуры в кандидатах, но overall top-K выигрывает у aux loss.

dVAE fixed-length и Residual KMeans заметно уступают VarLen dVAE-вариантам. Значит, в этих прогонах полезна не просто дискретизация embedding-ов, а именно learned VarLen SID с мягким metadata supervision.

## Эксперимент 1.2: структура SID и prefix purity

**Цель.** Проверить, становятся ли SID-префиксы более согласованными с artist/album metadata.

| Метод | Mean SID Length | Unique SIDs | Excess Collisions | Max Bucket Size | Artist Purity@1 | Album Purity@2 |
|---|---:|---:|---:|---:|---:|---:|
| dVAE fixed-length | 5.0000 | **263 537** | **4 517** | 40 | 0.2033 | **0.7332** |
| dVAE varlen | 3.2129 | 224 285 | 43 769 | 515 | 0.2042 | 0.7070 |
| dVAE varlen + aux loss | **2.3507** | 158 955 | 109 099 | 792 | 0.2055 | 0.4862 |
| dVAE varlen + prefix loss | 3.5421 | 219 089 | 48 965 | 744 | **0.2580** | 0.5040 |
| Residual KMeans | 5.0000 | 258 555 | 9 499 | 128 | 0.1378 | 0.4824 |
| Hier. artist/album R-KMeans | 5.0000 | 260 106 | 7 948 | **30** | 0.2147 | 0.5263 |

**Выводы.**

dVAE varlen + prefix loss действительно повышает Artist Purity@1: `0.2580` против `0.2042` у dVAE varlen. Это главный структурный успех prefix-level supervision.

При этом Album Purity@2 у metadata-aware dVAE вариантов ниже, чем у dVAE varlen: aux loss = `0.4862`, prefix loss = `0.5040`, dVAE varlen = `0.7070`. Вероятное объяснение — supervision по clustered/hashed labels и reconstruction loss конкурируют за короткий VarLen-код; префикс становится более artist-aware, но не обязательно лучше отражает raw album grouping.

У dVAE varlen + aux loss самый короткий средний SID (`2.35`), но и самые сильные коллизии: `109 099` excess collisions и max bucket size `792`. Downstream seqrec это частично компенсирует через `append_unique_token`, но структурно такой SID менее уникален.

Hierarchical artist/album R-KMeans имеет мало тяжелых collision buckets и самый маленький max bucket size (`30`), но его hard metadata structure не превращается в лучший overall Recall/NDCG. В этих результатах структурная интерпретируемость помогает не линейно: мягкий aux loss дает лучший ranking, а жесткий metadata baseline скорее помогает tail-части.

# Эксперимент 2: кому VarLen SID помогают сильнее

## Research question

**Дает ли VarLen dVAE больший прирост относительно fixed-length dVAE пользователям с длинной историей?**

Гипотеза: если у пользователя мало событий, модели почти не из чего понимать его вкус. Если событий много, variable-length semantic IDs могут быть полезнее, потому что они дают seqrec-модели более структурированное представление истории.

## Сетап

В этом эксперименте не используются artist/album-aware методы. Мы сравниваем только paper-style пару:

- `fixed`: fixed-length dVAE;
- `varlen`: VarLen dVAE.

Сравнение проводится на двух датасетах:

- Yambda;
- Amazon.

Для каждого датасета пользователи делятся на три группы по числу raw events в истории до target:

| Bin | Описание |
|---|---|
| short | нижняя треть пользователей по длине истории |
| medium | средняя треть |
| long | верхняя треть |

Важно: разбиение строится по raw event history до SID-токенизации. Поэтому один и тот же пользователь попадает в одну и ту же группу для `fixed` и `varlen`.

## Результаты

| Dataset | Method | History bin | Users | Mean history events | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Yambda | fixed | short | TBD | TBD | TBD | TBD | TBD | TBD |
| Yambda | varlen | short | TBD | TBD | TBD | TBD | TBD | TBD |
| Yambda | fixed | medium | TBD | TBD | TBD | TBD | TBD | TBD |
| Yambda | varlen | medium | TBD | TBD | TBD | TBD | TBD | TBD |
| Yambda | fixed | long | TBD | TBD | TBD | TBD | TBD | TBD |
| Yambda | varlen | long | TBD | TBD | TBD | TBD | TBD | TBD |
| Amazon | fixed | short | TBD | TBD | TBD | TBD | TBD | TBD |
| Amazon | varlen | short | TBD | TBD | TBD | TBD | TBD | TBD |
| Amazon | fixed | medium | TBD | TBD | TBD | TBD | TBD | TBD |
| Amazon | varlen | medium | TBD | TBD | TBD | TBD | TBD | TBD |
| Amazon | fixed | long | TBD | TBD | TBD | TBD | TBD | TBD |
| Amazon | varlen | long | TBD | TBD | TBD | TBD | TBD | TBD |

Отдельно нужно смотреть дельты `varlen - fixed`:

| Dataset | History bin | ΔRecall@10 | ΔNDCG@10 | ΔRecall@50 | ΔNDCG@50 |
|---|---|---:|---:|---:|---:|
| Yambda | short | TBD | TBD | TBD | TBD |
| Yambda | medium | TBD | TBD | TBD | TBD |
| Yambda | long | TBD | TBD | TBD | TBD |
| Amazon | short | TBD | TBD | TBD | TBD |
| Amazon | medium | TBD | TBD | TBD | TBD |
| Amazon | long | TBD | TBD | TBD | TBD |

**Выводы после финального запуска.**

TBD.

Что нужно обсудить:

- растет ли прирост `varlen - fixed` от `short` к `long`;
- согласован ли эффект на Yambda и Amazon;
- проявляется ли эффект сильнее в Recall или NDCG;
- есть ли датасет, где VarLen помогает только отдельной группе пользователей.

# Место для третьей части проекта

## Часть 3: [название третьего эксперимента]

TBD.

Кратко описать:

- цель;
- методы;
- основные результаты;
- вывод.

# Ограничения

- Используется один seed из-за compute budget.
- Seqrec-модель облегчена относительно paper-scale setup.
- Абсолютные Recall/NDCG нельзя напрямую сравнивать с результатами статьи.
- Artist/album metadata может быть шумной и many-to-many.
- Для dVAE supervision используются clustered/hashed labels, а raw labels применяются для structural evaluation.
- History-length эксперимент проверяет `fixed` vs `varlen`, но не доказывает, что любой semantic-ID метод будет вести себя так же.

# Финальное заключение

TBD после объединения всех частей проекта.

Финальный вывод должен коротко ответить:

- какой тип semantic IDs лучше сработал в recommendation;
- дала ли artist/album metadata пользу;
- кому VarLen SID помогают сильнее;
- насколько структурная интерпретируемость SID связана с downstream quality.
