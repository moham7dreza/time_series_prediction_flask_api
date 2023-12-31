"""Your migration message

Revision ID: ca60a0eaa9db
Revises: 3a6c549a5d3a
Create Date: 2023-12-28 16:04:17.887026

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ca60a0eaa9db'
down_revision = '3a6c549a5d3a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('predict',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('series', sa.String(length=255), nullable=False),
    sa.Column('models', sa.String(length=255), nullable=False),
    sa.Column('prices', sa.String(length=255), nullable=False),
    sa.Column('databases', sa.String(length=255), nullable=False),
    sa.Column('n_steps', sa.String(length=255), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('predict')
    # ### end Alembic commands ###
