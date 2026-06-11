import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# Preprocess the data
data['user_id'] = data['gender'] + "_" + data['age'] + "_" + data['destination']
coupon_mapping = {coupon: idx for idx, coupon in enumerate(data['coupon'].unique())}
item_to_coupon = {idx: coupon for coupon, idx in coupon_mapping.items()}
data['item_id'] = data['coupon'].map(coupon_mapping)
data['rating'] = data['Y']

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create user-item matrix for TRAINING data only
train_matrix = train_data.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

# Apply SVD
n_components = min(10, train_matrix.shape[0] - 1, train_matrix.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(train_matrix)
item_factors = svd.components_.T

# Reconstruct the rating matrix
predicted_ratings = np.dot(user_factors, item_factors.T)
pred_df = pd.DataFrame(predicted_ratings, index=train_matrix.index, columns=train_matrix.columns)

# Calculate RMSE on test set
rmse_values = []
for _, row in test_data.iterrows():
    user, item, actual = row['user_id'], row['item_id'], row['rating']
    if user in pred_df.index and item in pred_df.columns:
        pred = pred_df.loc[user, item]
        pred = max(0, min(1, pred))  # Clip
        rmse_values.append((actual - pred) ** 2)
    else:
        # If user/item not in training, use mean rating or 0
        pred = train_data['rating'].mean()
        rmse_values.append((actual - pred) ** 2)

rmse = np.sqrt(np.mean(rmse_values))
print(f"RMSE: {rmse:.4f}")

# Recommendation function
def recommend_coupons(user_id, n_recommendations=5):
    if user_id not in pred_df.index:
        print(f"User {user_id} not found in training data. Showing popular coupons.")
        # Fallback: Recommend most popular coupons
        popular_coupons = data['coupon'].value_counts().head(n_recommendations)
        for coupon, count in popular_coupons.items():
            print(f"  - {coupon} (Popularity: {count})")
        return

    user_predictions = pred_df.loc[user_id]
    user_train_data = train_data[train_data['user_id'] == user_id]
    rated_items = set(user_train_data['item_id'].values)

    recommendations = []
    for item_id, pred_rating in user_predictions.items():
        status = "Already Rated" if item_id in rated_items else "New"
        recommendations.append((item_id, pred_rating, status))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommendations[:n_recommendations]

    print(f"\n=== Top {n_recommendations} Coupon Recommendations for User: {user_id} ===")
    for item_id, pred_rating, status in top_recommendations:
        coupon_name = item_to_coupon.get(item_id, f"Coupon_{item_id}")
        print(f"  - {coupon_name} (Predicted Rating: {pred_rating:.4f}) [{status}]")

# Demo
sample_users = train_data['user_id'].unique()
if len(sample_users) > 0:
    sample_user = sample_users[0]
    print(f"\nSample user: {sample_user}")
    recommend_coupons(sample_user, n_recommendations=5)
