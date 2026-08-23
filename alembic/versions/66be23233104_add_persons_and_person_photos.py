"""add persons and person_photos

Revision ID: 66be23233104
Revises: ea3a397fdc6b
Create Date: 2026-08-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "66be23233104"
down_revision: str | Sequence[str] | None = "ea3a397fdc6b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "persons",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("relationship", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_persons_user_id_users"),
        sa.PrimaryKeyConstraint("id", name="pk_persons"),
    )
    op.create_index(op.f("ix_persons_user_id"), "persons", ["user_id"])

    op.create_table(
        "person_photos",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("person_id", sa.Uuid(), nullable=False),
        sa.Column("photo_storage_key", sa.String(length=512), nullable=False),
        sa.Column("photo_embedding", sa.LargeBinary(), nullable=False),
        sa.Column("face_roi", sa.JSON(), nullable=False),
        sa.Column("is_primary", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.ForeignKeyConstraint(
            ["person_id"], ["persons.id"], name="fk_person_photos_person_id_persons"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_person_photos"),
    )
    op.create_index(op.f("ix_person_photos_person_id"), "person_photos", ["person_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_person_photos_person_id"), table_name="person_photos")
    op.drop_table("person_photos")

    op.drop_index(op.f("ix_persons_user_id"), table_name="persons")
    op.drop_table("persons")
