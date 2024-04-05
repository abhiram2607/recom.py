import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Sample user-item interaction data
data = {
    'User': ['User1', 'User1', 'User2', 'User2', 'User3', 'User3', 'User4', 'User4'],
    'Item': ['Item1', 'Item2', 'Item2', 'Item3', 'Item1', 'Item3', 'Item2', 'Item4'],
    'Rating': [5, 4, 3, 5, 4, 2, 3, 1]
}
df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot(index='User', columns='Item', values='Rating').fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Function to get recommendations for a user
def get_recommendations(user):
    similar_users = user_similarity[user_item_matrix.index == user]
    similar_users_index = pd.Series(user_item_matrix.index, index=user_item_matrix.index)

    similar_users_index = similar_users_index.sort_values(ascending=False)

    recommendations = []
    for index, score in similar_users_index.iteritems():
        if index != user:
            user_interactions = user_item_matrix.loc[index]
            current_user_interactions = user_item_matrix.loc[user]

            new_items = user_interactions.index[user_interactions == 0]
            recommended_items = new_items[current_user_interactions[user_interactions.index] > 0]

            recommendations.extend(recommended_items)

    return list(set(recommendations))

# Get recommendations for User1
user_recommendations = get_recommendations('User1')
print(user_recommendations)
