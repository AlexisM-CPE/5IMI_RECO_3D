#include "points_detection.hpp"

using namespace cv;
using namespace std;

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r)
{
    float eps = 2.0f;       // We consider segments eps px longer to find more points
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;

    if (sqrt(pow(r.x-o1.x, 2) + pow(r.y-o1.y, 2)) <= sqrt(pow(p1.x-o1.x, 2) + pow(p1.y-o1.y, 2)) + eps && sqrt(pow(r.x-p1.x, 2) + pow(r.y-p1.y, 2)) <= sqrt(pow(p1.x-o1.x, 2) + pow(p1.y-o1.y, 2)) + eps &&
        sqrt(pow(r.x-o2.x, 2) + pow(r.y-o2.y, 2)) <= sqrt(pow(p2.x-o2.x, 2) + pow(p2.y-o2.y, 2)) + eps && sqrt(pow(r.x-p2.x, 2) + pow(r.y-p2.y, 2)) <= sqrt(pow(p2.x-o2.x, 2) + pow(p2.y-o2.y, 2)) + eps)
        return true;
    else
        return false;
        
}


vector<Point2f> merge_close_points(const vector<Point2f> &p, float r) {
	// Initialization of the weights
	vector<pair<Point2f, int>> points;
	for (int i = 0; i < p.size(); ++i) {
		points.push_back({p[i], 1});
	}

	// For each point
	vector<int> list_invalid;
	for (int i = 0; i < p.size(); ++i) {
		// if the point has already been visited
		if (find(list_invalid.begin(), list_invalid.end(), i) != list_invalid.end())
			continue;

		pair<Point2f, int> p1 = points[i];

		// We look for the points with a distance smaller than r
		for (int j = 0; j < p.size(); ++j) {
			// We discard the invalid points
			if (i == j || find(list_invalid.begin(), list_invalid.end(), j) != list_invalid.end())
				continue;
			pair<Point2f, int> p2 = points[j];

			// If the distance between p1 and p2 is less than r
			if (norm(p1.first-p2.first) <= r) {
				points[i].first = (p1.first*p1.second + p2.first*p2.second) / (p1.second + p2.second); // p1 et p2 fusionnent
				points[i].second += p2.second; // p1's weight increases
				list_invalid.push_back(j); // p2 is invalidated
				points[j].second = 0; // To clean up later
			}
		}
	}

	// Clean up
	vector<Point2f> ans;
	for (int i = 0; i < points.size(); ++i) {
		if (points[i].second != 0) {
			ans.push_back(points[i].first);
		}
	}

	return ans;
}