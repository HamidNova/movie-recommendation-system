import numpy as np


class HybridRecommender:

    def __init__(
        self,
        content_model,
        collab_model,
        df_ratings,
        df_movies
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.df_ratings = df_ratings
        self.df_movies = df_movies

        required_rating_columns = {
            "user_id",
            "movie_id",
        }

        missing_rating_columns = (
            required_rating_columns
            - set(df_ratings.columns)
        )

        if missing_rating_columns:
            raise ValueError(
                "df_ratings is missing required columns: "
                f"{missing_rating_columns}"
            )

        required_movie_columns = {
            "movie_id",
            "title",
        }

        missing_movie_columns = (
            required_movie_columns
            - set(df_movies.columns)
        )

        if missing_movie_columns:
            raise ValueError(
                "df_movies is missing required columns: "
                f"{missing_movie_columns}"
            )

        self.movie_lookup = (
            df_movies
            .set_index("movie_id")["title"]
            .to_dict()
        )

        self.title_to_movie_id = {}

        for _, row in df_movies.iterrows():

            title = str(row["title"]).strip().lower()
            movie_id = int(row["movie_id"])

            if (
                title
                and title not in self.title_to_movie_id
            ):
                self.title_to_movie_id[title] = movie_id

    @staticmethod
    def _validate_weight(content_weight):

        if not 0.0 <= content_weight <= 1.0:
            raise ValueError(
                "content_weight must be between 0 and 1."
            )

        return float(content_weight)

    def _get_movie_id_by_title(self, title):

        if title is None:
            return None

        normalized_title = (
            str(title)
            .strip()
            .lower()
        )

        if not normalized_title:
            return None

        return self.title_to_movie_id.get(
            normalized_title
        )

    def _get_rated_movie_ids(self, user_id):

        if user_id is None:
            return set()

        return set(
            self.df_ratings.loc[
                self.df_ratings["user_id"] == user_id,
                "movie_id"
            ].astype(int)
        )

    @staticmethod
    def _extract_score(rec):

        if not isinstance(rec, dict):
            return 0.0

        for key in [
            "predicted_rating",
            "similarity_score",
            "score",
            "rating",
            "confidence"
        ]:

            if key in rec:

                try:
                    return float(rec[key])

                except (
                    TypeError,
                    ValueError
                ):
                    continue

        return 0.0

    def _normalize_recommendations(
        self,
        recommendations
    ):

        if not recommendations:
            return []

        normalized_by_movie_id = {}

        for rec in recommendations:

            if not isinstance(rec, dict):
                continue

            movie_id = rec.get("movie_id")

            if movie_id is None:

                title = rec.get("title")

                if title:
                    movie_id = (
                        self._get_movie_id_by_title(
                            title
                        )
                    )

            if movie_id is None:
                continue

            try:
                movie_id = int(movie_id)

            except (
                TypeError,
                ValueError
            ):
                continue

            title = self.movie_lookup.get(
                movie_id
            )

            if title is None:
                continue

            score = self._extract_score(rec)

            normalized_rec = {
                "movie_id": movie_id,
                "title": str(title),
                "score": score
            }

            previous = normalized_by_movie_id.get(
                movie_id
            )

            if (
                previous is None
                or score > previous["score"]
            ):
                normalized_by_movie_id[
                    movie_id
                ] = normalized_rec

        return list(
            normalized_by_movie_id.values()
        )

    @staticmethod
    def _normalize_scores(
        recommendations
    ):

        if not recommendations:
            return {}

        scores = np.array(
            [
                rec["score"]
                for rec in recommendations
            ],
            dtype=np.float32
        )

        min_score = float(scores.min())
        max_score = float(scores.max())

        if max_score - min_score < 1e-8:

            return {
                rec["movie_id"]: 1.0
                for rec in recommendations
            }

        return {
            rec["movie_id"]: float(
                (rec["score"] - min_score)
                /
                (max_score - min_score)
            )
            for rec in recommendations
        }

    def _combine_recommendations(
        self,
        content_recs,
        collab_recs,
        n,
        content_weight,
        reason_content,
        reason_collab
    ):

        content_weight = self._validate_weight(
            content_weight
        )

        collab_weight = (
            1.0 - content_weight
        )

        content_scores = (
            self._normalize_scores(
                content_recs
            )
        )

        collab_scores = (
            self._normalize_scores(
                collab_recs
            )
        )

        content_ids = set(
            content_scores.keys()
        )

        collab_ids = set(
            collab_scores.keys()
        )

        all_movie_ids = (
            content_ids | collab_ids
        )

        hybrid_scores = {}

        for movie_id in all_movie_ids:

            score = 0.0

            if movie_id in content_scores:

                score += (
                    content_weight
                    * content_scores[movie_id]
                )

            if movie_id in collab_scores:

                score += (
                    collab_weight
                    * collab_scores[movie_id]
                )

            hybrid_scores[movie_id] = score

        sorted_movies = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        recommendations = []

        for movie_id, score in sorted_movies:

            title = self.movie_lookup.get(
                movie_id
            )

            if title is None:
                continue

            reasons = []

            if movie_id in content_ids:
                reasons.append(reason_content)

            if movie_id in collab_ids:
                reasons.append(reason_collab)

            recommendations.append({
                "movie_id": int(movie_id),
                "title": str(title),
                "confidence": round(
                    float(score),
                    4
                ),
                "predicted_rating": round(
                    float(score * 5.0),
                    3
                ),
                "reasons": reasons
            })

        return recommendations

    def _get_collaborative_recommendations(
        self,
        user_id,
        n
    ):

        if not hasattr(
            self.collab_model,
            "recommend_for_user"
        ):
            return []

        return (
            self.collab_model
            .recommend_for_user(
                user_id,
                self.df_ratings,
                self.df_movies,
                n=n
            )
            or []
        )

    def recommend_by_movie_id(
        self,
        user_id,
        movie_id,
        n=5,
        content_weight=0.4
    ):

        if n <= 0:
            return []

        content_weight = self._validate_weight(
            content_weight
        )

        movie_id = int(movie_id)

        if movie_id not in self.movie_lookup:
            return self._format_collaborative_only(
                self._get_collaborative_recommendations(
                    user_id=user_id,
                    n=n
                ),
                n
            )

        fetch_n = max(
            n * 3,
            20
        )

        rated_movie_ids = (
            self._get_rated_movie_ids(
                user_id
            )
        )

        content_recs = []

        if hasattr(
            self.content_model,
            "recommend_by_movie_id"
        ):

            content_recs = (
                self.content_model
                .recommend_by_movie_id(
                    movie_id,
                    n=fetch_n
                )
                or []
            )

        collab_recs = (
            self._get_collaborative_recommendations(
                user_id=user_id,
                n=fetch_n
            )
        )

        content_recs = (
            self._normalize_recommendations(
                content_recs
            )
        )

        collab_recs = (
            self._normalize_recommendations(
                collab_recs
            )
        )

        excluded_movie_ids = (
            rated_movie_ids | {movie_id}
        )

        content_recs = [
            rec
            for rec in content_recs
            if rec["movie_id"]
            not in excluded_movie_ids
        ]

        collab_recs = [
            rec
            for rec in collab_recs
            if rec["movie_id"]
            not in excluded_movie_ids
        ]

        return self._combine_recommendations(
            content_recs=content_recs,
            collab_recs=collab_recs,
            n=n,
            content_weight=content_weight,
            reason_content=(
                "similar to your selected movie"
            ),
            reason_collab=(
                "recommended based on similar users"
            )
        )

    def recommend(
        self,
        user_id,
        liked_movie_title,
        n=5,
        content_weight=0.4
    ):

        if n <= 0:
            return []

        if liked_movie_title is None:

            return self._format_collaborative_only(
                self._get_collaborative_recommendations(
                    user_id=user_id,
                    n=n
                ),
                n
            )

        search_title = (
            str(liked_movie_title)
            .strip()
            .lower()
        )

        if not search_title:

            return self._format_collaborative_only(
                self._get_collaborative_recommendations(
                    user_id=user_id,
                    n=n
                ),
                n
            )

        exact_movie_id = (
            self.title_to_movie_id.get(
                search_title
            )
        )

        if exact_movie_id is None:

            mask = (
                self.df_movies["title"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_title,
                    na=False,
                    regex=False
                )
            )

            if mask.any():

                exact_movie_id = int(
                    self.df_movies.loc[
                        mask,
                        "movie_id"
                    ].iloc[0]
                )

        if exact_movie_id is not None:

            return self.recommend_by_movie_id(
                user_id=user_id,
                movie_id=exact_movie_id,
                n=n,
                content_weight=content_weight
            )

        collab_recs = (
            self._get_collaborative_recommendations(
                user_id=user_id,
                n=n
            )
        )

        return self._format_collaborative_only(
            collab_recs,
            n
        )

    def _format_collaborative_only(
        self,
        recommendations,
        n
    ):

        normalized = (
            self._normalize_recommendations(
                recommendations
            )
        )

        result = []

        for rec in normalized[:n]:

            score = float(rec["score"])

            result.append({
                "movie_id": int(
                    rec["movie_id"]
                ),
                "title": rec["title"],
                "confidence": round(
                    score,
                    4
                ),
                "predicted_rating": round(
                    score,
                    3
                ),
                "reasons": [
                    "recommended based on similar users"
                ]
            })

        return result

    def recommend_similar_movies(
        self,
        movie_id,
        n=5,
        content_weight=0.6
    ):

        if n <= 0:
            return []

        content_weight = self._validate_weight(
            content_weight
        )

        movie_id = int(movie_id)

        if movie_id not in self.movie_lookup:
            return []

        fetch_n = max(
            n * 3,
            20
        )

        content_recs = []

        if hasattr(
            self.content_model,
            "recommend_by_movie_id"
        ):

            content_recs = (
                self.content_model
                .recommend_by_movie_id(
                    movie_id,
                    n=fetch_n
                )
                or []
            )

        collab_recs = []

        if hasattr(
            self.collab_model,
            "recommend_similar_movies"
        ):

            collab_recs = (
                self.collab_model
                .recommend_similar_movies(
                    movie_id,
                    self.df_movies,
                    n=fetch_n
                )
                or []
            )

        content_recs = (
            self._normalize_recommendations(
                content_recs
            )
        )

        collab_recs = (
            self._normalize_recommendations(
                collab_recs
            )
        )

        content_recs = [
            rec
            for rec in content_recs
            if rec["movie_id"] != movie_id
        ]

        collab_recs = [
            rec
            for rec in collab_recs
            if rec["movie_id"] != movie_id
        ]

        return self._combine_recommendations(
            content_recs=content_recs,
            collab_recs=collab_recs,
            n=n,
            content_weight=content_weight,
            reason_content=(
                "similar genres and movie content"
            ),
            reason_collab=(
                "similar according to user behavior"
            )
        )
