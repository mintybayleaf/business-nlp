# duckdb_cosine.py
import duckdb
import numpy as np
import uuid
from typing import Iterator, Tuple, Optional

# Change ":memory:" to "demo.duckdb" to persist to disk
CONNECTION = duckdb.connect(database=":memory:")


def setup_table(table_name: str, vector_length: int) -> None:
    """
    Create a table for storing embeddings as DOUBLE[vector_length].
    id is a TEXT primary key (we generate UUIDs on insert).
    """
    CONNECTION.execute(f"DROP TABLE IF EXISTS {table_name}")
    CONNECTION.execute(
        f"""
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            name TEXT,
            embedding DOUBLE[{vector_length}]
        )
        """
    )


def insert_np_array(table_name: str, name: str, array: np.ndarray, row_id: Optional[str] = None) -> str:
    """
    Insert a numpy array as an embedding. Returns the row id (UUID).
    Ensures the array is float64 to match DOUBLE[...] column type.
    """
    if row_id is None:
        row_id = str(uuid.uuid4())
    arr = np.asarray(array, dtype=np.float64)
    # parameterized insert: DuckDB accepts Python lists for DOUBLE[] columns
    CONNECTION.execute(
        f"INSERT INTO {table_name} (id, name, embedding) VALUES (?, ?, ?)",
        (row_id, name, arr.tolist()),
    )
    return row_id


def cosine_distance_nearest_vectors(
    table_name: str,
    query_array: np.ndarray,
    total: int = 3,
    clamp_eps: float = 1e-6,
    round_digits: int = 4,
) -> Iterator[Tuple[str, float]]:
    """
    Yields (name, distance) for the nearest `total` vectors by cosine distance.
    - Converts query_array to float64 (DOUBLE).
    - Uses array_cosine_distance(embedding, ?::DOUBLE[n]) to avoid type mismatch.
    - Clamps tiny floating noise to 0.0 and rounds the result.
    """
    arr = np.asarray(query_array, dtype=np.float64)
    n = arr.size
    if n == 0:
        return

    # We cast the parameter into DOUBLE[n] to match the stored embedding type exactly.
    # Note: table_name is interpolated — ensure it's trusted or sanitize if needed.
    query = f"""
        SELECT name, array_cosine_distance(embedding, ?::DOUBLE[{n}]) AS distance
        FROM {table_name}
        ORDER BY distance
        LIMIT {int(total)}
    """
    results = CONNECTION.execute(query, [arr.tolist()]).fetchall()

    for name, distance in results:
        if distance is None:
            continue
        # clamp tiny values to zero to avoid scientific notation tiny noise
        if abs(distance) < clamp_eps:
            distance = 0.0
        yield (name, round(float(distance), round_digits))


if __name__ == "__main__":
    # --- Example usage ---
    setup_table("my_vectors", 3)

    insert_np_array("my_vectors", "vec1", np.array([0.1, 0.2, 0.3]))
    insert_np_array("my_vectors", "vec2", np.array([0.5, 0.6, 0.7]))
    insert_np_array("my_vectors", "vec3", np.array([0.9, 0.8, 0.4]))
    insert_np_array("my_vectors", "twitter", np.array([0.01, 0.02, 0.03]))
    insert_np_array("my_vectors", "twitter corp emporium san antonio", np.array([0.2, 0.1, 0.0]))
    insert_np_array("my_vectors", "twitter san diego", np.array([0.01, 0.02, 0.03]))  # near-identical to "twitter"

    # Query example
    query_vec = np.array([0.01, 0.02, 0.03])
    print("[cosine distance] query='twitter san diego'")
    for name, dist in cosine_distance_nearest_vectors("my_vectors", query_vec, total=5):
        print((name, dist))
