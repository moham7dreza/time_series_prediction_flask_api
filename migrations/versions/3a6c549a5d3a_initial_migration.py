"""Initial migration

Revision ID: 3a6c549a5d3a
Revises: 
Create Date: 2023-12-28 15:06:25.162932

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3a6c549a5d3a'
down_revision = None
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