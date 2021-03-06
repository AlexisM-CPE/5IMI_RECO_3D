#include "points_detection.hpp"

using namespace cv;
using namespace std;

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r)
{
	float eps = 2.0f; // We consider segments eps px longer to find more points
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < /*EPS*/ 1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;

	if (sqrt(pow(r.x - o1.x, 2) + pow(r.y - o1.y, 2)) <= sqrt(pow(p1.x - o1.x, 2) + pow(p1.y - o1.y, 2)) + eps && sqrt(pow(r.x - p1.x, 2) + pow(r.y - p1.y, 2)) <= sqrt(pow(p1.x - o1.x, 2) + pow(p1.y - o1.y, 2)) + eps &&
		sqrt(pow(r.x - o2.x, 2) + pow(r.y - o2.y, 2)) <= sqrt(pow(p2.x - o2.x, 2) + pow(p2.y - o2.y, 2)) + eps && sqrt(pow(r.x - p2.x, 2) + pow(r.y - p2.y, 2)) <= sqrt(pow(p2.x - o2.x, 2) + pow(p2.y - o2.y, 2)) + eps)
		return true;
	else
		return false;
}

vector<Point2f> merge_close_points(const vector<Point2f> &p, float r)
{
	// Initialization of the weights
	vector<pair<Point2f, int>> points;
	for (unsigned int i = 0; i < p.size(); ++i)
	{
		points.push_back({p[i], 1});
	}

	// For each point
	vector<int> list_invalid;
	for (unsigned int i = 0; i < p.size(); ++i)
	{
		// if the point has already been visited
		if (find(list_invalid.begin(), list_invalid.end(), i) != list_invalid.end())
			continue;

		pair<Point2f, int> p1 = points[i];

		// We look for the points with a distance smaller than r
		for (unsigned int j = 0; j < p.size(); ++j)
		{
			// We discard the invalid points
			if (i == j || find(list_invalid.begin(), list_invalid.end(), j) != list_invalid.end())
				continue;
			pair<Point2f, int> p2 = points[j];

			// If the distance between p1 and p2 is less than r
			if (norm(p1.first - p2.first) <= r)
			{
				points[i].first = (p1.first * p1.second + p2.first * p2.second) / (p1.second + p2.second); // p1 et p2 fusionnent
				points[i].second += p2.second;															   // p1's weight increases
				list_invalid.push_back(j);																   // p2 is invalidated
				points[j].second = 0;																	   // To clean up later
			}
		}
	}

	// Clean up
	vector<Point2f> ans;
	for (unsigned int i = 0; i < points.size(); ++i)
	{
		if (points[i].second != 0)
		{
			ans.push_back(points[i].first);
		}
	}

	return ans;
}

vector<int> kmeans(vector<float> const &angles, int it_max, int k, float min_v, float max_v)
{
	//float mean_base = mean()

	float dist_to_zero = 0.0f;
	for (int i = 0; i < angles.size(); i++)
	{
		float dist = angles[i];
		dist_to_zero += dist;
	}

	dist_to_zero = dist_to_zero / angles.size();

	float mean_test_1;
	float mean_test_2;
	float mean_1 = fmod(dist_to_zero - CV_PI / 4.0f, /*2*/ CV_PI);
	float mean_2 = fmod(dist_to_zero + CV_PI / 4.0f, /*2*/ CV_PI);
	vector<float> new_angles;
	for (int i = 0; i < angles.size(); i++)
	{
		new_angles.push_back(fmod(angles[i], CV_PI));
	}

	// Initialization of the labels and the distances
	vector<int> labels(new_angles.size(), 0);
	vector<float> dist(new_angles.size(), 10.0f);

	int it = 0;
	while (it < it_max)
	{
		mean_test_1 = mean_1;
		mean_test_2 = mean_2;

		vector<float> sum_val(2, 0.0f);
		vector<float> nb_elem(2, 0.0f);

		for (unsigned int j = 0; j < new_angles.size(); j++)
		{
			vector<float> dist_j;

			float dist_1 = abs(mean_test_1 - new_angles[j]);
			float dist_2 = abs(mean_test_2 - new_angles[j]);

			if (dist_1 > CV_PI / 2)
				dist_1 = CV_PI - dist_1;

			if (dist_2 > CV_PI / 2)
				dist_2 = CV_PI - dist_2;

			if (dist_1 < dist_2)
			{
				labels[j] = 0;
				nb_elem[0] += 1;
				if (new_angles[j] > mean_test_1)
					sum_val[0] += dist_1;
				else
					sum_val[0] -= dist_1;
			}
			else
			{
				labels[j] = 1;
				nb_elem[1] += 1;
				if (new_angles[j] > mean_test_2)
					sum_val[1] += dist_2;
				else
					sum_val[1] -= dist_2;
			}
		}

		// New means
		mean_1 += sum_val[0] / nb_elem[0];
		mean_2 += sum_val[1] / nb_elem[1];
		it++;
	}

	return labels;
}

vector<float> get_angles(vector<Vec4i> linesP)
{
	vector<float> angles;
	for (unsigned int i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		int x0 = l[0];
		int y0 = l[1];
		int x1 = l[2];
		int y1 = l[3];

		Vec2f vhoriz = Vec2f(1.0f, 0.0f);
		Vec2f line;

		if (y0 >= y1)
		{
			line = Vec2f(float(x0 - x1), float(y0 - y1));
		}
		else
		{
			line = Vec2f(float(x1 - x0), float(y1 - y0));
		}

		float norm_v = sqrt(pow(line[0], 2) + pow(line[1], 2)) + 0.00001f;
		line = line / norm_v;
		float cos_a = vhoriz[0] * line[0] + vhoriz[1] * line[1]; //dot(&vhoriz, &line);

		float cross_product_z = vhoriz[1] * line[0] - vhoriz[0] * line[1];
		float angle = acos(cos_a); //fmod(acos(cos_a), CV_PI); //acos(cos_a); //fmod(acos(cos_a), CV_PI);

		if (cross_product_z < 0)
		{
			angle = CV_PI - angle;
		}
		angles.push_back(angle);
	}
	return angles;
}
