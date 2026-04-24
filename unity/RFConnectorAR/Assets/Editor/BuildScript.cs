#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace RFConnectorAR.EditorTools
{
    /// <summary>
    /// Command-line-invocable build entry points. Call these via:
    ///
    ///     Unity.exe -batchmode -projectPath &lt;path&gt; \
    ///         -buildTarget Android \
    ///         -executeMethod RFConnectorAR.EditorTools.BuildScript.BuildAndroid \
    ///         -quit -nographics
    ///
    ///     Unity.exe -batchmode -projectPath &lt;path&gt; \
    ///         -buildTarget iOS \
    ///         -executeMethod RFConnectorAR.EditorTools.BuildScript.BuildIOS \
    ///         -quit -nographics
    ///
    /// Output lands under &lt;project&gt;/Builds/Android/RFConnectorAR.apk
    /// or &lt;project&gt;/Builds/iOS/ (an Xcode project).
    /// </summary>
    public static class BuildScript
    {
        private static string[] EnabledScenes =>
            new[]
            {
                "Assets/Scenes/Scanner.unity",
                "Assets/Scenes/Enroll.unity",
                "Assets/Scenes/Curate.unity",
            };

        public static void BuildAndroid()
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(
                BuildTargetGroup.Android, BuildTarget.Android);

            // Android minimum-viable player settings (tuned in Plan 2 Task 1).
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel24;
            PlayerSettings.SetScriptingBackend(
                NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;

            string outDir = Path.Combine(
                Path.GetDirectoryName(Application.dataPath)!, "Builds", "Android");
            Directory.CreateDirectory(outDir);
            string outPath = Path.Combine(outDir, "RFConnectorAR.apk");

            var opts = new BuildPlayerOptions
            {
                scenes = EnabledScenes,
                locationPathName = outPath,
                target = BuildTarget.Android,
                options = BuildOptions.None,
            };

            BuildReport report = BuildPipeline.BuildPlayer(opts);
            Report(report, "Android", outPath);
            if (report.summary.result != BuildResult.Succeeded)
            {
                EditorApplication.Exit(1);
            }
        }

        public static void BuildIOS()
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(
                BuildTargetGroup.iOS, BuildTarget.iOS);

            // iOS player settings from the spec.
            PlayerSettings.iOS.targetOSVersionString = "13.0";
            PlayerSettings.iOS.cameraUsageDescription =
                "Scans RF connectors for identification.";

            string outDir = Path.Combine(
                Path.GetDirectoryName(Application.dataPath)!, "Builds", "iOS");
            Directory.CreateDirectory(outDir);

            var opts = new BuildPlayerOptions
            {
                scenes = EnabledScenes,
                locationPathName = outDir,
                target = BuildTarget.iOS,
                options = BuildOptions.None,
            };

            BuildReport report = BuildPipeline.BuildPlayer(opts);
            Report(report, "iOS", outDir);
            if (report.summary.result != BuildResult.Succeeded)
            {
                EditorApplication.Exit(1);
            }
        }

        private static void Report(BuildReport report, string label, string outPath)
        {
            var summary = report.summary;
            Debug.Log(
                $"BuildScript: {label} build result={summary.result} " +
                $"sizeBytes={summary.totalSize} duration={summary.totalTime} " +
                $"output={outPath}");
        }
    }
}
#endif
