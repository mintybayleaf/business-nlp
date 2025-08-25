import psycopg2


CONNECTION = psycopg2.connect(
    dbname="demo", user="postgres", password="postgres", host="localhost", port=5432
)


def setup_table(name, vector_length):
    cursor = CONNECTION.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {name}")
    cursor.execute(
        f"""CREATE TABLE IF NOT EXISTS {name} (
            id SERIAL PRIMARY KEY,
            name TEXT,
            embedding vector({vector_length})
        );"""
    )
    CONNECTION.commit()


def insert_np_array(table_name, name, array):
    cursor = CONNECTION.cursor()
    cursor.execute(
        f"INSERT INTO {table_name} (name, embedding) VALUES (%s, %s::vector)",
        (name, array.tolist()),
    )
    CONNECTION.commit()


def cosine_distance_nearest_vectors(table_name, array, total=3):
    cursor = CONNECTION.cursor()
    cursor.execute(
        f"""
        SELECT name, embedding <=> %s::vector AS distance
        FROM {table_name}
        ORDER BY distance
        LIMIT {total};
        """,
        (array.tolist(),),
    )

    for row in cursor.fetchall():
        yield row
