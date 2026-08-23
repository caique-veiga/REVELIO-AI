"""drop detected_objects (YOLO pipeline removed)

Revision ID: 9f4c1b2a7d3e
Revises: 66be23233104
Create Date: 2026-08-23 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "9f4c1b2a7d3e"
down_revision: str | Sequence[str] | None = "66be23233104"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index(op.f("ix_detected_objects_scene_id"), table_name="detected_objects")
    op.drop_table("detected_objects")


def downgrade() -> None:
    op.create_table(
        "detected_objects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("scene_id", sa.Uuid(), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(length=100), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox_x1", sa.Integer(), nullable=False),
        sa.Column("bbox_y1", sa.Integer(), nullable=False),
        sa.Column("bbox_x2", sa.Integer(), nullable=False),
        sa.Column("bbox_y2", sa.Integer(), nullable=False),
        sa.Column("position_horizontal", sa.String(length=10), nullable=False),
        sa.Column("position_vertical", sa.String(length=10), nullable=False),
        sa.Column("position_region", sa.String(length=20), nullable=False),
        sa.Column("color_name", sa.String(length=50), nullable=False),
        sa.Column("color_r", sa.Integer(), nullable=False),
        sa.Column("color_g", sa.Integer(), nullable=False),
        sa.Column("color_b", sa.Integer(), nullable=False),
        sa.Column("color_confidence", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["scene_id"], ["scenes.id"], name="fk_detected_objects_scene_id_scenes"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_detected_objects"),
    )
    op.create_index(op.f("ix_detected_objects_scene_id"), "detected_objects", ["scene_id"])
