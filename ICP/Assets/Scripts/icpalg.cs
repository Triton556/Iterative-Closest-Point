using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Supercluster.KDTree;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Diagnostics;
using System.Linq;
using Random = System.Random;

namespace MyNamespace
{
    public class icpalg : MonoBehaviour
    {
        [SerializeField] private GameObject MGameObject;
        [SerializeField] private GameObject PGameObject;

        private Mesh mPointsMesh;
        private Mesh pPointsMesh;

        private float threshHold = 0.0f;

        public bool startICP = false;

        public Matrix<double> ParceFromVector3(Mesh mesh, GameObject gameObject)
        {
            List<Vector3> vector3Verts = new List<Vector3>();
            mesh.GetVertices(vector3Verts);
            Matrix<double> m = Matrix<double>.Build.Dense(3, vector3Verts.Count, 0);

            for (int i = 0; i < vector3Verts.Count; i++)
            {
                m[0, i] = vector3Verts[i].x + gameObject.transform.position.x;
                m[1, i] = vector3Verts[i].y + gameObject.transform.position.y;
                m[2, i] = vector3Verts[i].z + gameObject.transform.position.z;
            }

            return m;
        }

        public Matrix<double> ParceFromVector3Local(Mesh mesh)
        {
            List<Vector3> vector3Verts = new List<Vector3>();
            mesh.GetVertices(vector3Verts);
            Matrix<double> m = Matrix<double>.Build.Dense(3, vector3Verts.Count, 0);

            for (int i = 0; i < vector3Verts.Count; i++)
            {
                m[0, i] = vector3Verts[i].x;
                m[1, i] = vector3Verts[i].y;
                m[2, i] = vector3Verts[i].z;
            }

            return m;
        }

        /*public Vector3[] ParceFromMatrix(Matrix<double> vertexMatrix)
        {
            Vector3[] vector3Verts = new Vector3[vertexMatrix.ColumnCount];

            for (int i = 0; i < vector3Verts.Length; i++)
            {
                vector3Verts[i].x = (float) vertexMatrix[0, i];
                vector3Verts[i].y = (float) vertexMatrix[1, i];
                vector3Verts[i].z = (float) vertexMatrix[2, i];
            }

            print("Converted from matrix to verts");
            return vector3Verts;
        }

        public Vector3 ParceFromMatrix(Vector<double> vertexMatrix)
        {
            Vector3 vector3Verts = new Vector3();


            vector3Verts.x = (float) vertexMatrix[0];
            vector3Verts.y = (float) vertexMatrix[1];
            vector3Verts.z = (float) vertexMatrix[2];


            print("Converted from matrix to verts");
            return vector3Verts;
        }
        */


        public static Tuple<Vector<double>, Matrix<double>, double> ICP_run(Matrix<double> M_points,
            Matrix<double> P_points, Matrix<double> M_points_local)
        {
            int Np = P_points.ColumnCount;
            var m = Matrix<double>.Build;

            Matrix<double> Y;
            Y = KD_tree(M_points, P_points);
            double s = 1;

            Matrix<double> R;
            Matrix<double> t;
            Vector<double> d;
            double err = 0;
            Matrix<double> dummy_Row = m.Dense(1, Np, 0);

            Matrix<double> Mu_p = FindCentroid(P_points);
            //Matrix<double> Mu_m = FindCentroid(M_points);
            Matrix<double> Mu_y = FindCentroid(Y);

            Matrix<double> dummy_p1 = m.Dense(1, Np);
            Matrix<double> dummy_p2 = m.Dense(1, Np);
            Matrix<double> dummy_p3 = m.Dense(1, Np, 0);
            Matrix<double> dummy_y1 = m.Dense(1, Np);
            Matrix<double> dummy_y2 = m.Dense(1, Np);
            Matrix<double> dummy_y3 = m.Dense(1, Np, 0);

            dummy_p1.SetRow(0, P_points.Row(0));
            dummy_p2.SetRow(0, P_points.Row(1));
            dummy_p3.SetRow(0, P_points.Row(2));

            Matrix<double> P_prime =
                (dummy_p1 - Mu_p[0, 0]).Stack((dummy_p2 - Mu_p[1, 0]).Stack(dummy_p3 - Mu_p[2, 0]));

            dummy_y1.SetRow(0, Y.Row(0));
            dummy_y2.SetRow(0, Y.Row(1));
            dummy_y3.SetRow(0, Y.Row(2));

            Matrix<double> Y_prime =
                (dummy_y1 - Mu_y[0, 0]).Stack((dummy_y2 - Mu_y[1, 0]).Stack(dummy_y3 - Mu_y[2, 0]));

            Matrix<double> Px = m.Dense(1, Np);
            Matrix<double> Py = m.Dense(1, Np);
            Matrix<double> Pz = m.Dense(1, Np, 0);
            Matrix<double> Yx = m.Dense(1, Np);
            Matrix<double> Yy = m.Dense(1, Np);
            Matrix<double> Yz = m.Dense(1, Np, 0);

            Px.SetRow(0, P_prime.Row(0));
            Py.SetRow(0, P_prime.Row(1));
            Pz.SetRow(0, P_prime.Row(2));
            Yx.SetRow(0, Y_prime.Row(0));
            Yy.SetRow(0, Y_prime.Row(1));
            Yz.SetRow(0, Y_prime.Row(2));

            var Sxx = Px * Yx.Transpose();
            var Sxy = Px * Yx.Transpose();
            var Sxz = Px * Yx.Transpose();

            var Syx = Px * Yx.Transpose();
            var Syy = Px * Yx.Transpose();
            var Syz = Px * Yx.Transpose();

            var Szx = Px * Yx.Transpose();
            var Szy = Px * Yx.Transpose();
            var Szz = Px * Yx.Transpose();

            Matrix<double> Nmatrix = m.DenseOfArray(new[,]
            {
                {
                    Sxx[0, 0] + Syy[0, 0] + Szz[0, 0], Syz[0, 0] - Szy[0, 0], -Sxz[0, 0] + Szx[0, 0],
                    Sxy[0, 0] - Syx[0, 0]
                },
                {
                    -Szy[0, 0] + Syz[0, 0], Sxx[0, 0] - Syy[0, 0] - Szz[0, 0], Sxy[0, 0] + Syx[0, 0],
                    Sxz[0, 0] + Szx[0, 0]
                },
                {
                    Szx[0, 0] - Sxz[0, 0], Syx[0, 0] + Sxy[0, 0], -Sxx[0, 0] + Syy[0, 0] - Szz[0, 0],
                    Syz[0, 0] + Szy[0, 0]
                },
                {
                    -Syx[0, 0] + Sxy[0, 0], Szx[0, 0] + Sxz[0, 0], Szy[0, 0] + Syz[0, 0],
                    -Sxx[0, 0] + Szz[0, 0] - Syy[0, 0]
                }
            });

            var evd = Nmatrix.Evd();
            Matrix<double> eigenvectors = evd.EigenVectors;
            var q = eigenvectors.Column(3);
            var q0 = q[0];
            var q1 = q[1];
            var q2 = q[2];
            var q3 = q[3];

            var Qbar = m.DenseOfArray(new[,]
            {
                {q0, -q1, -q2, -q3},
                {q1, q0, q3, -q2},
                {q2, -q3, q0, q1},
                {q3, q2, -q1, q0}
            });

            var Q = m.DenseOfArray(new[,]
            {
                {q0, -q1, -q2, -q3},
                {q1, q0, -q3, q2},
                {q2, q3, q0, -q1},
                {q3, -q2, q1, q0}
            });


            R = (Qbar.Transpose()).Multiply(Q);
            R = (R.RemoveColumn(0)).RemoveRow(0);


            //t = Mu_y + R * Mu_p;
            var Mu_p_local = FindCentroid(M_points_local);
            t = Mu_y - s * R * Mu_p_local;

            for (int i = 0; i < Np; i++)
            {
                d = Y.Column(i).Subtract(P_points.Column(i));
                err += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                print(err);
            }

            //Tuple<Matrix<double>, Matrix<double>, double> ret =
            //    new Tuple<Matrix<double>, Matrix<double>, double>(R, t, err);

            Tuple<Vector<double>, Matrix<double>, double> ret =
                new Tuple<Vector<double>, Matrix<double>, double>(q, t, err);

            return ret;
        }

        public static Matrix<double> KD_tree(Matrix<double> M_points, Matrix<double> P_points)
        {
            int Np = P_points.ColumnCount;

            Func<double[], double[], double> L2Norm = (x, y) =>
            {
                double dist = 0;

                for (int i = 0; i < x.Length; i++)
                {
                    dist += (x[i] - y[i]) * (x[i] - y[i]);
                }

                return dist;
            };


            var treeData2 = M_points.ToColumnArrays();

            //print(treeData2[0][0]);
            //print(treeData2[0][1]);
            //print(treeData2[0][2]);
            //print(treeData2[0][3]);
            //print(treeData2[0][4]);
            //print(treeData2[0][5]);


            var treeNodes = treeData2.Select(p => p.ToString()).ToArray();

            var m = Matrix<double>.Build;
            Matrix<double> Y = m.Dense(3, Np, 0);
            Tuple<double[], string>[] test;
            var tree = new KDTree<double, string>(3, treeData2, treeNodes, L2Norm);

            var scan_data = P_points.ToColumnArrays();

            for (int i = 0; i < scan_data.Length; i++)
            {
                test = tree.NearestNeighbors(scan_data[i], 1);
                //print(test);
                Y[0, i] = test[0].Item1[0];
                Y[1, i] = test[0].Item1[1];
                Y[2, i] = test[0].Item1[2];
            }

            return Y;
        }


        public static Matrix<double> FindCentroid(Matrix<double> points)
        {
            var m = Matrix<double>.Build;
            int column_count = points.ColumnCount;
            Matrix<double> centroid = m.Dense(3, 1, 0);
            double TotalX = 0;
            double TotalY = 0;
            double TotalZ = 0;

            double AvrX = 0;
            double AvrY = 0;
            double AvrZ = 0;

            for (int i = 0; i < column_count; i++)
            {
                TotalX += points[0, i];
                TotalY += points[1, i];
                TotalZ += points[2, i];
            }

            AvrX = TotalX / column_count;
            AvrY = TotalY / column_count;
            AvrZ = TotalZ / column_count;

            centroid[0, 0] = AvrX;
            centroid[1, 0] = AvrY;
            centroid[2, 0] = AvrZ;

            return centroid;
        }

        //public static Matrix<double> ApplyMove(Tuple<Matrix<double>, Matrix<double>, double> ret, Matrix<double> P_points)
        /*
        public static Matrix<double> ApplyMove(Tuple<Matrix<double>, Matrix<double>, double> ret,
            Matrix<double> P_points)
        {
            var R = ret.Item1;
            var t = ret.Item2;
            var err = ret.Item3;

            var P_points2 = R.Multiply(P_points);

            var m = Matrix<double>.Build;
            var Np = P_points.ColumnCount;

            Vector<double> d;

            Matrix<double> Px2 = m.Dense(1, Np);
            Matrix<double> Py2 = m.Dense(1, Np);
            Matrix<double> Pz2 = m.Dense(1, Np);


            Px2.SetRow(0, P_points2.Row(0));
            Py2.SetRow(0, P_points2.Row(1));
            Pz2.SetRow(0, P_points2.Row(2));
            Px2 = Px2 + t[0, 0];
            Py2 = Py2 + t[1, 0];
            Pz2 = Pz2 + t[2, 0];
            P_points.SetRow(0, Px2.Row(0));
            P_points.SetRow(1, Py2.Row(0));
            P_points.SetRow(2, Pz2.Row(0));


            return P_points;
        }
        */

        // Start is called before the first frame update
        void Start()
        {
            mPointsMesh = MGameObject.GetComponent<MeshFilter>().mesh;
            pPointsMesh = PGameObject.GetComponent<MeshFilter>().mesh;
            //print(pPointsMesh.GetBaseVertex(0));
        }

        // Update is called once per frame
        void Update()
        {
            if (startICP)
            {
                mPointsMesh = MGameObject.GetComponent<MeshFilter>().mesh;
                pPointsMesh = PGameObject.GetComponent<MeshFilter>().mesh;
                var MPointsMatrix = ParceFromVector3(mPointsMesh, MGameObject);
                var PPointsMatrix = ParceFromVector3(pPointsMesh, PGameObject);
                var MPointsMatrixLocal = ParceFromVector3Local(mPointsMesh);
                var Rte = ICP_run(MPointsMatrix, PPointsMatrix, MPointsMatrixLocal);

                if (Rte.Item3 > threshHold)
                {
                    //var newPPointsMatrix = ApplyMove(Rte, PPointsMatrix);

                    //pPointsMesh.SetVertices(ParceFromMatrix(newPPointsMatrix));

                    //PGameObject.transform.rotation =
                    //    new Quaternion((float)Rte.Item1[1], (float)Rte.Item1[2], (float)Rte.Item1[3], (float)Rte.Item1[0]);

                    if (PGameObject.transform.rotation != MGameObject.transform.rotation)
                    {
                        PGameObject.transform.rotation =
                            new Quaternion((float) Rte.Item1[1], (float) Rte.Item1[2], (float) Rte.Item1[3],
                                (float) Rte.Item1[0]);
                    }


                    PGameObject.transform.position =
                        new Vector3((float) Rte.Item2[0, 0], (float) Rte.Item2[1, 0], (float) Rte.Item2[2, 0]);

                    //print(Rte.Item3);

                    //PGameObject.transform.position = new Vector3((float)Rte.Item2[0, 0], (float)Rte.Item2[1, 0], (float)Rte.Item2[2, 0]);

                    startICP = false;
                }
            }
        }
    }
}